from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
from collections import defaultdict
import time

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter


@dataclass
class UsageRecord:
    timestamp: datetime
    operation_type: str
    tokens_used: int
    user_id: str
    duration_ms: float
    status: str
    metadata: Optional[Dict] = None


class TelemetryService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._usage_records: List[UsageRecord] = []
        self._user_totals = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

        # Initialize OpenTelemetry
        resource = Resource.create({"service.name": "databridge-core"})

        # Initialize tracing
        trace_provider = TracerProvider(resource=resource)
        otlp_span_exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(otlp_span_exporter)
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__)

        # Initialize metrics
        metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(metric_provider)
        self.meter = metrics.get_meter(__name__)

        # Create metrics
        self.operation_counter = self.meter.create_counter(
            "databridge.operations",
            description="Number of operations performed",
        )
        self.token_counter = self.meter.create_counter(
            "databridge.tokens",
            description="Number of tokens processed",
        )
        self.operation_duration = self.meter.create_histogram(
            "databridge.operation.duration",
            description="Duration of operations",
            unit="ms",
        )

    async def track_operation(
        self,
        operation_type: str,
        user_id: str,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Context manager for tracking operations with both usage metrics and OpenTelemetry
        """
        start_time = time.time()
        status = "success"
        current_span = trace.get_current_span()

        try:
            # Add operation attributes to the current span
            current_span.set_attribute("operation.type", operation_type)
            current_span.set_attribute("user.id", user_id)
            if metadata:
                for key, value in metadata.items():
                    current_span.set_attribute(f"metadata.{key}", str(value))

            # Record metrics
            self.operation_counter.add(1, {"operation": operation_type, "user_id": user_id})
            if tokens_used > 0:
                self.token_counter.add(
                    tokens_used, {"operation": operation_type, "user_id": user_id}
                )

            yield current_span

        except Exception as e:
            status = "error"
            current_span.set_status(Status(StatusCode.ERROR))
            current_span.record_exception(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.operation_duration.record(
                duration_ms, {"operation": operation_type, "user_id": user_id, "status": status}
            )

            # Record usage
            record = UsageRecord(
                timestamp=datetime.utcnow(),
                operation_type=operation_type,
                tokens_used=tokens_used,
                user_id=user_id,
                duration_ms=duration_ms,
                status=status,
                metadata=metadata,
            )

            with self._lock:
                self._usage_records.append(record)
                self._user_totals[user_id][operation_type] += tokens_used
                self._user_totals[user_id]["total"] += tokens_used

    def get_user_usage(self, user_id: str) -> Dict[str, int]:
        """
        Get usage statistics for a specific user
        """
        with self._lock:
            return dict(self._user_totals[user_id])

    def get_recent_usage(
        self,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> List[UsageRecord]:
        """
        Get recent usage records with filtering options
        """
        with self._lock:
            records = self._usage_records
            if user_id:
                records = [r for r in records if r.user_id == user_id]
            if operation_type:
                records = [r for r in records if r.operation_type == operation_type]
            if since:
                records = [r for r in records if r.timestamp >= since]
            if status:
                records = [r for r in records if r.status == status]
            return records
