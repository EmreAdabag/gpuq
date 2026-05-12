from __future__ import annotations

import os
import shlex
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import config as config_mod
from . import daemon as daemon_mod
from . import jobs as jobs_mod
from . import launch as launch_mod
from . import paths, workers
from .jobs import Job, now_iso, next_id, write_job, state_lock


app = typer.Typer(add_completion=False, no_args_is_help=True, help="Submit GPU jobs over SSH.")
workers_app = typer.Typer(add_completion=False, invoke_without_command=True, help="Manage workers.")
app.add_typer(workers_app, name="workers")

console = Console()


def _load_or_die() -> config_mod.Config:
    try:
        return config_mod.load_config()
    except FileNotFoundError:
        console.print(f"[red]No config at {paths.config_path()}.[/red]")
        console.print("Run [bold]gpuq workers add localhost[/bold] to get started.")
        raise typer.Exit(2)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def submit(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Job name (default: job-<id>)"),
    host: Optional[str] = typer.Option(None, "--host", help="Pin job to a specific worker host"),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPUs required"),
):
    """Submit a job. Use `--` to separate gpuq flags from the command.

    Example: gpuq submit -n myrun -- python train.py --config foo.yaml
    """
    _load_or_die()
    extra = list(ctx.args)
    if not extra:
        console.print("[red]Provide a command after `--`.[/red]")
        raise typer.Exit(2)
    cmd_str = " ".join(shlex.quote(a) for a in extra)
    with state_lock():
        jid = next_id()
        job = Job(
            id=jid,
            name=name or f"job-{jid}",
            command=cmd_str,
            status="queued",
            host_pin=host,
            gpus_requested=gpus,
            submitted_at=now_iso(),
        )
        write_job(job)
    console.print(f"Submitted job [bold]{jid}[/bold] ({job.name})")


@app.command("ps")
def ps_cmd(
    all_: bool = typer.Option(False, "--all", "-a", help="Include terminal states"),
):
    """List jobs (default: queued + running)."""
    rows = jobs_mod.list_jobs()
    if not all_:
        rows = [j for j in rows if j.status in ("queued", "running")]
    if not rows:
        console.print("[dim]No jobs.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", justify="right")
    table.add_column("NAME")
    table.add_column("STATUS")
    table.add_column("HOST")
    table.add_column("GPUS")
    table.add_column("SUBMITTED")
    table.add_column("CMD", max_width=60, overflow="ellipsis")
    for j in rows:
        gpus_str = (
            ",".join(str(g) for g in j.gpus_assigned)
            if j.gpus_assigned
            else f"req={j.gpus_requested}"
        )
        table.add_row(
            str(j.id),
            j.name,
            j.status,
            j.host or "-",
            gpus_str,
            j.submitted_at or "-",
            j.command,
        )
    console.print(table)


@app.command()
def logs(
    id: int = typer.Argument(...),
    follow: bool = typer.Option(False, "-f", "--follow"),
):
    """Tail a job's log file (SSH-tailed on the worker)."""
    job = jobs_mod.try_read_job(id)
    if not job:
        console.print(f"[red]Job {id} not found.[/red]")
        raise typer.Exit(2)
    if not job.log_path:
        console.print(f"[yellow]Job {id} has no log yet (still queued?).[/yellow]")
        raise typer.Exit(0)
    cfg = _load_or_die()
    worker = cfg.worker(job.host or "")
    if not worker:
        console.print(f"[red]Job's host {job.host} not in config.[/red]")
        raise typer.Exit(2)
    follow_flag = "-F" if follow else ""
    ssh_args = ["ssh"]
    if worker.ssh_key:
        ssh_args += ["-i", worker.ssh_key]
    ssh_args += [
        worker.ssh_target,
        f"tail -n 200 {follow_flag} {shlex.quote(job.log_path)}",
    ]
    os.execvp("ssh", ssh_args)


@app.command()
def attach(id: int = typer.Argument(...)):
    """ssh -t into the worker and attach the job's tmux session."""
    job = jobs_mod.try_read_job(id)
    if not job:
        console.print(f"[red]Job {id} not found.[/red]")
        raise typer.Exit(2)
    if job.status != "running":
        console.print(f"[red]Job {id} not running (status={job.status}).[/red]")
        raise typer.Exit(2)
    cfg = _load_or_die()
    worker = cfg.worker(job.host or "")
    if not worker:
        console.print(f"[red]Host {job.host} not in config.[/red]")
        raise typer.Exit(2)
    ssh_args = ["ssh", "-t"]
    if worker.ssh_key:
        ssh_args += ["-i", worker.ssh_key]
    ssh_args += [worker.ssh_target, f"tmux attach -t {shlex.quote(job.session_name)}"]
    os.execvp("ssh", ssh_args)


@app.command()
def kill(id: int = typer.Argument(...)):
    """Kill a queued or running job."""
    job = jobs_mod.try_read_job(id)
    if not job:
        console.print(f"[red]Job {id} not found.[/red]")
        raise typer.Exit(2)
    if job.status in ("done", "failed", "killed"):
        console.print(f"[yellow]Job {id} already terminal ({job.status}).[/yellow]")
        return
    with state_lock():
        # Re-read inside lock to avoid racing the daemon's dispatch.
        fresh = jobs_mod.try_read_job(id) or job
        if fresh.status == "queued":
            fresh.status = "killed"
            fresh.ended_at = now_iso()
            write_job(fresh)
            console.print(f"Killed queued job {id}.")
            return
    jobs_mod.request_kill(id)
    console.print(f"Requested kill of running job {id}; daemon will handle.")


@workers_app.callback()
def workers_root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        _workers_list()


@workers_app.command("list")
def workers_list_cmd():
    """List configured workers and their GPU state."""
    _workers_list()


def _workers_list() -> None:
    cfg = _load_or_die()
    all_jobs = jobs_mod.list_jobs()
    if not cfg.workers:
        console.print("[dim]No workers configured. Run `gpuq workers add <host>`.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("HOST")
    table.add_column("USER")
    table.add_column("ALLOWED GPUS")
    table.add_column("GPU MEM USED (MB)")
    table.add_column("RUNNING JOBS")
    for w in cfg.workers:
        try:
            stats = workers.probe_gpus(w, timeout=10)
            mem_str = ", ".join(f"{s.index}:{s.used_mb}" for s in stats) or "-"
        except Exception as e:
            mem_str = f"[red]err: {e}[/red]"
        running = [str(j.id) for j in all_jobs if j.status == "running" and j.host == w.host]
        table.add_row(
            w.host,
            w.user,
            ",".join(str(g) for g in w.gpus),
            mem_str,
            ",".join(running) if running else "-",
        )
    console.print(table)


@workers_app.command("add")
def workers_add(host: str = typer.Argument(...)):
    """Interactively onboard a worker."""
    cfg_path = paths.config_path()
    if not cfg_path.exists():
        config_mod.write_default_config(cfg_path)
        console.print(f"Wrote default config at {cfg_path}")

    default_user = os.environ.get("USER") or "root"
    user = typer.prompt("SSH user", default=default_user)
    ssh_key_in = typer.prompt("SSH key (blank = ssh default)", default="").strip()
    ssh_key = os.path.expanduser(ssh_key_in) if ssh_key_in else None
    candidate = config_mod.WorkerConfig(host=host, user=user, gpus=[], ssh_key=ssh_key)

    console.print(f"Testing SSH to {candidate.ssh_target}...")
    res = workers.run_ssh(candidate, "true", timeout=15)
    if res.returncode != 0:
        console.print(f"[red]SSH failed: {res.stderr.strip() or 'unknown error'}[/red]")
        console.print("Hint: set up key auth (`ssh-copy-id`) and try again.")
        raise typer.Exit(2)

    console.print("Probing GPUs...")
    res = workers.run_ssh(
        candidate,
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    )
    if res.returncode != 0:
        console.print(f"[red]nvidia-smi failed: {res.stderr.strip()}[/red]")
        raise typer.Exit(2)
    console.print(res.stdout)
    indices: list[int] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            indices.append(int(parts[0]))

    cfg_now = config_mod.load_config()
    sm = cfg_now.shared_mount
    console.print(f"Checking shared mount {sm}...")
    if not workers.remote_test_dir(candidate, sm):
        console.print(f"[red]Shared mount {sm} not present on {host}.[/red]")
        if not typer.confirm("Continue anyway?", default=False):
            raise typer.Exit(2)

    for bin_ in ("tmux", "uv"):
        path = workers.remote_which(candidate, bin_)
        if path:
            console.print(f"  {bin_}: {path}")
        else:
            console.print(f"[yellow]{bin_} not found on {host}.[/yellow]")
            if bin_ == "uv":
                console.print("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
            elif bin_ == "tmux":
                console.print("Install with your package manager (apt install tmux).")
            if not typer.confirm(f"Continue without {bin_}?", default=False):
                raise typer.Exit(2)

    default_gpus = ",".join(str(i) for i in indices)
    chosen = typer.prompt("GPUs gpuq may use", default=default_gpus)
    candidate.gpus = [int(x) for x in chosen.split(",") if x.strip()]
    config_mod.add_worker_to_config(candidate)
    console.print(f"[green]Added {host} to config.[/green]")


@workers_app.command("refresh")
def workers_refresh_cmd():
    """Re-probe all workers."""
    _workers_list()


@app.command()
def sync(host: str = typer.Argument(...)):
    """rsync the local repo and run uv sync on a worker. Useful for warming hosts."""
    cfg = _load_or_die()
    w = cfg.worker(host)
    if not w:
        console.print(f"[red]Unknown host {host}.[/red]")
        raise typer.Exit(2)
    repo_path = f"{cfg.remote_repo_base}/warmup"
    workers.remote_mkdir_p(w, repo_path)
    launch_mod.rsync_repo(cfg, w, repo_path)
    launch_mod.update_last_synced(w, cfg, repo_path)
    rc = launch_mod.uv_sync(w, repo_path, "/dev/null")
    if rc != 0:
        console.print(f"[red]uv sync exited {rc}.[/red]")
        raise typer.Exit(rc)
    console.print(f"[green]Synced repo and env to {host}.[/green]")


@app.command()
def daemon(
    foreground: bool = typer.Option(False, "--foreground", help="Don't fork; log to stderr."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start the gpuq daemon."""
    daemon_mod.run_daemon(foreground=foreground, verbose=verbose)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
