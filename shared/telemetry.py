"""
shared/telemetry.py
====================
Khoi tao OpenTelemetry tracer dung chung cho moi service.

Moi service goi setup_tracing(service_name) mot lan duy nhat khi startup,
sau do dung get_tracer() de lay tracer va bat dau span.

OTLP exporter gui trace ve Jaeger collector qua gRPC (port 4317).
Neu OTEL_EXPORTER_OTLP_ENDPOINT chua set hoac exporter loi, tracer
fallback ve NoOpTracer -- service van chay binh thuong, chi mat trace.

Chi can cai them:
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
    Khoi tao TracerProvider va export OTLP ve Jaeger.
    Goi mot lan duy nhat trong lifespan hoac module-level cua moi service.

    app: FastAPI instance, neu truyen vao thi FastAPIInstrumentor se tu dong
         tao span cho moi HTTP request ma khong can them decorator tay.
    """
    global _tracer

    if not OTEL_AVAILABLE:
        print(f"[telemetry:{service_name}] opentelemetry chua install -- tracing disabled.")
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
        print(f"[telemetry:{service_name}] Setup loi: {e} -- tracing disabled.")
        _tracer = None


def get_tracer():
    """
    Tra ve tracer da khoi tao, hoac NoOpTracer neu chua setup.
    Caller dung nhu sau:
        with get_tracer().start_as_current_span("ten_span") as span:
            span.set_attribute("key", "value")
    """
    if not OTEL_AVAILABLE or _tracer is None:
        return _NoOpTracer()
    return _tracer


class _NoOpSpan:
    """Span gia -- khong lam gi, chi de code goi set_attribute ma khong crash."""

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
    """Tracer gia khi OTEL chua install hoac exporter loi."""

    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()

    def start_span(self, name, **kwargs):
        return _NoOpSpan()
