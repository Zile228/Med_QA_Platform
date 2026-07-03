"""
shared/telemetry.py

Shared OpenTelemetry tracer initialization used by every service.

Each service calls setup_tracing(service_name) exactly once on startup,
then uses get_tracer() to get the tracer and start spans.

The OTLP exporter sends traces to the Jaeger collector over gRPC (port 4317).
If OTEL_EXPORTER_OTLP_ENDPOINT is unset or the exporter fails, the tracer
falls back to NoOpTracer -- the service still runs normally, it just loses tracing.

Requires installing:
    opentelemetry-sdk
    opentelemetry-exporter-otlp-proto-grpc
    opentelemetry-instrumentation-fastapi
    opentelemetry-instrumentation-httpx
"""

import os

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

_tracer = None


def setup_tracing(service_name: str, app=None) -> None:
    """
    Initialize the TracerProvider and export OTLP to Jaeger.
    Call exactly once in the lifespan or module-level of each service.

    app: FastAPI instance, if provided FastAPIInstrumentor will automatically
         create a span for every HTTP request without needing manual decorators.
    """
    global _tracer

    if not OTEL_AVAILABLE:
        print(f"[telemetry:{service_name}] opentelemetry not installed -- tracing disabled.")
        return

    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://jaeger:4317",
    )

    try:
        resource = Resource.create({"service.name": service_name})
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)

        if app is not None:
            FastAPIInstrumentor.instrument_app(app)
            HTTPXClientInstrumentor().instrument()

        print(f"[telemetry:{service_name}] Tracing -> {endpoint}")
    except Exception as e:
        print(f"[telemetry:{service_name}] Setup failed: {e} -- tracing disabled.")
        _tracer = None


def get_tracer():
    """
    Return the initialized tracer, or NoOpTracer if not yet set up.
    Callers use it like this:
        with get_tracer().start_as_current_span("span_name") as span:
            span.set_attribute("key", "value")
    """
    if not OTEL_AVAILABLE or _tracer is None:
        return _NoOpTracer()
    return _tracer


class _NoOpSpan:
    """Fake span -- does nothing, just lets code call set_attribute without crashing."""

    def set_attribute(self, key, value):
        pass

    def record_exception(self, exc):
        pass

    def set_status(self, status):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """Fake tracer used when OTEL is not installed or the exporter fails."""

    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()

    def start_span(self, name, **kwargs):
        return _NoOpSpan()
