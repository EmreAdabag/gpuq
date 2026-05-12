from __future__ import annotations

from gpuq.workers import parse_nvidia_smi, select_available_gpus


def test_parse_nvidia_smi_basic() -> None:
    out = "0, 412\n1, 23012\n2, 0\n3, 8\n"
    stats = parse_nvidia_smi(out)
    assert [(s.index, s.used_mb) for s in stats] == [(0, 412), (1, 23012), (2, 0), (3, 8)]


def test_parse_nvidia_smi_ignores_garbage() -> None:
    out = "0, 412\n\nfoo\n2, x\n3, 5\n"
    stats = parse_nvidia_smi(out)
    assert [(s.index, s.used_mb) for s in stats] == [(0, 412), (3, 5)]


def test_select_available_all_free() -> None:
    avail = select_available_gpus(
        allowed=[0, 1, 2, 3],
        gpu_stats={0: 12, 1: 30, 2: 5, 3: 100},
        used_by_gpuq=set(),
        threshold_mb=500,
    )
    assert avail == [0, 1, 2, 3]


def test_select_filters_busy_by_memory() -> None:
    # GPU 1 is heavily used by some external process -> exclude
    avail = select_available_gpus(
        allowed=[0, 1, 2],
        gpu_stats={0: 10, 1: 20000, 2: 30},
        used_by_gpuq=set(),
        threshold_mb=500,
    )
    assert avail == [0, 2]


def test_select_excludes_gpus_already_used_by_gpuq() -> None:
    avail = select_available_gpus(
        allowed=[0, 1, 2, 3],
        gpu_stats={0: 12, 1: 30, 2: 5, 3: 100},
        used_by_gpuq={1, 2},
        threshold_mb=500,
    )
    assert avail == [0, 3]


def test_select_respects_allowlist() -> None:
    avail = select_available_gpus(
        allowed=[0, 1],
        gpu_stats={0: 10, 1: 10, 2: 10, 3: 10},
        used_by_gpuq=set(),
        threshold_mb=500,
    )
    assert avail == [0, 1]  # 2 and 3 excluded though physically free


def test_select_skips_indices_missing_from_smi() -> None:
    # Hot-unplugged GPU or driver hiccup: index not in smi output -> skip.
    avail = select_available_gpus(
        allowed=[0, 1, 2],
        gpu_stats={0: 0, 2: 0},
        used_by_gpuq=set(),
        threshold_mb=500,
    )
    assert avail == [0, 2]


def test_select_sorted_low_to_high() -> None:
    avail = select_available_gpus(
        allowed=[3, 1, 0, 2],
        gpu_stats={0: 0, 1: 0, 2: 0, 3: 0},
        used_by_gpuq=set(),
        threshold_mb=500,
    )
    assert avail == [0, 1, 2, 3]
