from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoopTelemetry(ProductTelemetryClient):
    """Drop Chroma product telemetry.

    Chroma 0.6 still calls PostHog when telemetry is disabled, and current
    PostHog releases reject that call. The demo does not send usage events.
    """

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
