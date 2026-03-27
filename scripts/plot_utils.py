#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSample:
    timestamp: float
    active_stage: str | None
    metrics: dict[tuple[str, tuple[tuple[str, str], ...]], float]


def parse_metric_key(raw_key: str) -> tuple[str, dict[str, str]]:
    if "{" not in raw_key:
        return raw_key, {}
    name, raw_labels = raw_key.split("{", 1)
    labels = json.loads("{" + raw_labels)
    return name, {str(k): str(v) for k, v in labels.items()}


def load_samples(path: Path) -> list[MetricSample]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing metrics samples at {path}") from exc

    samples: list[MetricSample] = []
    for row in payload:
        parsed_metrics: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        for raw_key, raw_value in row.get("metrics", {}).items():
            metric_name, labels = parse_metric_key(raw_key)
            label_items = tuple(sorted(labels.items()))
            parsed_metrics[(metric_name, label_items)] = float(raw_value)
        samples.append(
            MetricSample(
                timestamp=float(row["timestamp"]),
                active_stage=row.get("active_stage"),
                metrics=parsed_metrics,
            )
        )
    return samples


def load_summary(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing benchmark summary at {path}") from exc


def ensure_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")


def relative_times(samples: list[MetricSample]) -> list[float]:
    if not samples:
        return []
    start_time = samples[0].timestamp
    return [sample.timestamp - start_time for sample in samples]


def metric_value(
    sample: MetricSample,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    if labels is None:
        labels = {}
    candidate_value: float | None = None
    best_extra_labels = 10**9
    for (name, label_items), value in sample.metrics.items():
        if name != metric_name:
            continue
        label_dict = dict(label_items)
        if any(label_dict.get(k) != v for k, v in labels.items()):
            continue
        extra_labels = len(label_dict) - len(labels)
        if extra_labels < best_extra_labels:
            candidate_value = value
            best_extra_labels = extra_labels
    return candidate_value


def metric_series(
    samples: list[MetricSample],
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = metric_value(sample, metric_name, labels=labels)
        values.append(float("nan") if value is None else float(value))
    return values


def stage_segments(samples: list[MetricSample], times: list[float]) -> list[tuple[float, float, str]]:
    if not samples or not times:
        return []
    segments: list[tuple[float, float, str]] = []
    current_stage = samples[0].active_stage or "unknown"
    start_t = times[0]
    for sample, t in zip(samples[1:], times[1:]):
        stage = sample.active_stage or "unknown"
        if stage != current_stage:
            segments.append((start_t, t, current_stage))
            current_stage = stage
            start_t = t
    segments.append((start_t, times[-1], current_stage))
    return segments


def metric_max_across_labels(sample: MetricSample, metric_name: str) -> float | None:
    values = [float(value) for (name, _labels), value in sample.metrics.items() if name == metric_name]
    if not values:
        return None
    return max(values)
