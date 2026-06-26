"""Workload providers for the GrADyS RL bridge showcase.

The Alibaba loader intentionally uses the cluster trace only as a task-feature
source. Spatial locations, UAV mobility, wireless links, and service windows
remain owned by the GrADyS environment.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple


@dataclass(frozen=True)
class WorkloadSample:
    compute_demand: float
    data_size: float
    deadline_delta: float


class WorkloadProvider(Protocol):
    max_compute_demand: float
    max_data_size: float

    def sample(self, rng: random.Random) -> WorkloadSample:
        pass


class SyntheticWorkload:
    """Uniform synthetic workload used for quick tests and ablations."""

    def __init__(
        self,
        deadline_range: Tuple[float, float],
        compute_demand_range: Tuple[float, float],
        data_size_range: Tuple[float, float],
    ) -> None:
        self.deadline_range = deadline_range
        self.compute_demand_range = compute_demand_range
        self.data_size_range = data_size_range
        self.max_compute_demand = compute_demand_range[1]
        self.max_data_size = data_size_range[1]

    def sample(self, rng: random.Random) -> WorkloadSample:
        return WorkloadSample(
            compute_demand=rng.uniform(*self.compute_demand_range),
            data_size=rng.uniform(*self.data_size_range),
            deadline_delta=rng.uniform(*self.deadline_range),
        )


class AlibabaClusterV2017Workload:
    """Trace-derived task-feature sampler for Alibaba Cluster Data V2017.

    Expected inputs are CSV files such as `batch_task.csv`. The parser accepts
    either a header row or the common v2017 column order:

    task_name, instance_num, job_name, task_type, status, start_time, end_time,
    plan_cpu, plan_mem

    The resulting sample fields are intentionally normalized to the simplified
    UAV-MEC environment:

    - `compute_demand`: duration * requested CPU, then rescaled.
    - `data_size`: requested memory used as a payload proxy, then rescaled.
    - `deadline_delta`: trace task duration, clipped into the configured range.
    """

    DEFAULT_COLUMNS = [
        "task_name",
        "instance_num",
        "job_name",
        "task_type",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
    ]

    def __init__(
        self,
        samples: List[WorkloadSample],
        fallback: SyntheticWorkload,
    ) -> None:
        if not samples:
            raise ValueError("Alibaba workload loader produced zero valid samples.")
        self.samples = samples
        self.fallback = fallback
        self.max_compute_demand = max(sample.compute_demand for sample in samples)
        self.max_data_size = max(sample.data_size for sample in samples)

    @classmethod
    def from_csv(
        cls,
        path: str,
        max_rows: int,
        deadline_range: Tuple[float, float],
        compute_demand_range: Tuple[float, float],
        data_size_range: Tuple[float, float],
    ) -> "AlibabaClusterV2017Workload":
        fallback = SyntheticWorkload(deadline_range, compute_demand_range, data_size_range)
        rows = _read_rows(Path(path), max_rows=max_rows)
        raw_features = [
            _row_to_feature(row)
            for row in rows
        ]
        raw_features = [feature for feature in raw_features if feature is not None]
        if not raw_features:
            raise ValueError(f"No usable task rows found in {path}.")

        raw_compute = [feature["raw_compute"] for feature in raw_features]
        raw_memory = [feature["raw_memory"] for feature in raw_features]

        samples = [
            WorkloadSample(
                compute_demand=_rescale(
                    feature["raw_compute"],
                    min(raw_compute),
                    max(raw_compute),
                    compute_demand_range,
                ),
                data_size=_rescale(
                    feature["raw_memory"],
                    min(raw_memory),
                    max(raw_memory),
                    data_size_range,
                ),
                deadline_delta=_clip(feature["duration"], *deadline_range),
            )
            for feature in raw_features
        ]
        return cls(samples, fallback)

    def sample(self, rng: random.Random) -> WorkloadSample:
        return rng.choice(self.samples)


def _read_rows(path: Path, max_rows: int) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as file_obj:
        sample = file_obj.read(4096)
        file_obj.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader: Iterable[Dict[str, str]] = csv.DictReader(file_obj)
        else:
            reader = csv.DictReader(file_obj, fieldnames=AlibabaClusterV2017Workload.DEFAULT_COLUMNS)

        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append(row)
    return rows


def _row_to_feature(row: Dict[str, str]) -> Optional[Dict[str, float]]:
    start_time = _first_float(row, ["start_time", "start", "start_timestamp"])
    end_time = _first_float(row, ["end_time", "end", "end_timestamp"])
    plan_cpu = _first_float(row, ["plan_cpu", "cpu", "cpu_request", "request_cpu"])
    plan_mem = _first_float(row, ["plan_mem", "mem", "memory", "memory_request", "request_mem"])

    if start_time is None or end_time is None:
        return None
    duration = end_time - start_time
    if duration <= 0 or not math.isfinite(duration):
        return None

    cpu = plan_cpu if plan_cpu is not None and plan_cpu > 0 else 1.0
    memory = plan_mem if plan_mem is not None and plan_mem > 0 else 1.0
    return {
        "duration": duration,
        "raw_compute": duration * cpu,
        "raw_memory": memory,
    }


def _first_float(row: Dict[str, str], names: List[str]) -> Optional[float]:
    lower = {key.lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lower.get(name.lower())
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "null", "none"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _rescale(value: float, source_min: float, source_max: float, target: Tuple[float, float]) -> float:
    low, high = target
    if source_max <= source_min:
        return (low + high) / 2.0
    ratio = (value - source_min) / (source_max - source_min)
    return low + ratio * (high - low)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
