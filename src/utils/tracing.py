"""
Simbay OpenTelemetry tracing integration.

All interaction with the OTel SDK is isolated in this module.
When SIMBAY_OTEL_ENABLED is not set to "1", "true", "yes", or "on",
every function in this module is a documented no-op. The rest of the
codebase never imports opentelemetry directly.
"""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from typing import Iterator
from typing import Any


def _is_enabled() -> bool:
    return os.getenv("SIMBAY_OTEL_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def _otel_endpoint() -> str:
    return os.getenv("SIMBAY_OTEL_ENDPOINT", "http://tempo:4317")


def setup_tracing(run_id: str) -> None:
    if not _is_enabled():
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": "simbay",
            "service.version": "0.1.0",
            "run_id": run_id,
            "simbay.run_id": run_id,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=_otel_endpoint(),
                insecure=True,
            ),
            max_queue_size=32768,
            max_export_batch_size=1024,
            schedule_delay_millis=1000,
        )
    )
    trace.set_tracer_provider(provider)


def force_flush_tracing() -> None:
    if not _is_enabled():
        return

    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    force_flush_fn = getattr(provider, "force_flush", None)
    if callable(force_flush_fn):
        force_flush_fn()


def shutdown_tracing() -> None:
    if not _is_enabled():
        return

    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    force_flush_fn = getattr(provider, "force_flush", None)
    if callable(force_flush_fn):
        force_flush_fn()
    shutdown_fn = getattr(provider, "shutdown", None)
    if callable(shutdown_fn):
        shutdown_fn()


def get_tracer(name: str):
    from opentelemetry import trace

    return trace.get_tracer(name)


@contextmanager
def span(
    tracer,
    name: str,
    _attributes: dict[str, str | int | float | bool] | None = None,
) -> Iterator[Any]:
    with tracer.start_as_current_span(name) as current_span:
        yield current_span


def add_exemplar(run_id: str, step: int) -> dict[str, str]:
    if not _is_enabled():
        return {}

    from opentelemetry import trace

    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if not ctx.is_valid:
        return {}
    return {
        "traceID": format(ctx.trace_id, "032x"),
        "spanID": format(ctx.span_id, "016x"),
        "run_id": run_id,
        "step": str(step),
    }


def set_span_attributes(attributes: dict[str, str | int | float | bool | None]) -> None:
    if not _is_enabled():
        return

    from opentelemetry import trace

    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if not ctx.is_valid:
        return
    for key, value in attributes.items():
        if value is not None:
            current_span.set_attribute(key, value)


def trace_call(
    tracer_name: str,
    span_name: str | None = None,
):
    def decorator(func):
        resolved_span_name = span_name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(tracer_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with span(tracer, resolved_span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_public_methods(
    tracer_name: str,
    *,
    include_private: bool = False,
    exclude: set[str] | None = None,
):
    excluded = set(exclude or set())

    def decorator(cls):
        for name, value in list(cls.__dict__.items()):
            if name in excluded:
                continue
            if name.startswith("__") and name.endswith("__") and name != "__init__":
                continue
            if not include_private and name.startswith("_") and name != "__init__":
                continue
            if not inspect.isfunction(value):
                continue
            setattr(
                cls,
                name,
                trace_call(
                    tracer_name,
                    span_name=f"{cls.__module__}.{cls.__name__}.{name}",
                )(value),
            )
        return cls

    return decorator
