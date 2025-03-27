from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
from collections import defaultdict
import time
from contextlib import asynccontextmanager
import os
import json
from pathlib import Path
import uuid
import hashlib
import logging

from core.config import get_settings

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    MetricExporter,
    AggregationTemporality,
    MetricsData,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
import requests
from urllib3.exceptions import ProtocolError, ReadTimeoutError

# Get settings from config
settings = get_settings()

# Telemetry configuration - use settings directly from TOML
TELEMETRY_ENABLED = settings.TELEMETRY_ENABLED
HONEYCOMB_ENABLED = settings.HONEYCOMB_ENABLED

# Honeycomb configuration - using proxy to avoid exposing API key in code
# Default to localhost:8080 for the proxy, but allow override from settings
HONEYCOMB_PROXY_ENDPOINT = getattr(settings, "HONEYCOMB_PROXY_ENDPOINT", "https://otel-proxy.onrender.com")
HONEYCOMB_PROXY_ENDPOINT = HONEYCOMB_PROXY_ENDPOINT if isinstance(HONEYCOMB_PROXY_ENDPOINT, str) and len(HONEYCOMB_PROXY_ENDPOINT) > 0 else "https://otel-proxy.onrender.com"
SERVICE_NAME = settings.SERVICE_NAME

# Headers for OTLP - no API key needed as the proxy will add it
OTLP_HEADERS = {
    "Content-Type": "application/x-protobuf"
}

# Configure timeouts and retries directly from TOML config
OTLP_TIMEOUT = settings.OTLP_TIMEOUT
OTLP_MAX_RETRIES = settings.OTLP_MAX_RETRIES
OTLP_RETRY_DELAY = settings.OTLP_RETRY_DELAY
OTLP_MAX_EXPORT_BATCH_SIZE = settings.OTLP_MAX_EXPORT_BATCH_SIZE
OTLP_SCHEDULE_DELAY_MILLIS = settings.OTLP_SCHEDULE_DELAY_MILLIS
OTLP_MAX_QUEUE_SIZE = settings.OTLP_MAX_QUEUE_SIZE

# OTLP endpoints - using our proxy instead of direct Honeycomb connection
OTLP_TRACES_ENDPOINT = f"{HONEYCOMB_PROXY_ENDPOINT}/v1/traces"
OTLP_METRICS_ENDPOINT = f"{HONEYCOMB_PROXY_ENDPOINT}/v1/metrics"

# Enable debug logging for OpenTelemetry
os.environ["OTEL_PYTHON_LOGGING_LEVEL"] = "INFO"  # Changed from DEBUG to reduce verbosity
# Add export protocol setting if not already set
if not os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL"):
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

def get_installation_id() -> str:
    """Generate or retrieve a unique anonymous installation ID."""
    id_file = Path.home() / ".databridge" / "installation_id"
    id_file.parent.mkdir(parents=True, exist_ok=True)
    
    if id_file.exists():
        return id_file.read_text().strip()
    
    # Generate a new installation ID
    # We hash the machine-id (if available) or a random UUID
    machine_id_file = Path("/etc/machine-id")
    if machine_id_file.exists():
        machine_id = machine_id_file.read_text().strip()
    else:
        machine_id = str(uuid.uuid4())
    
    # Hash the machine ID to make it anonymous
    installation_id = hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    
    # Save it for future use
    id_file.write_text(installation_id)
    return installation_id


class FileSpanExporter:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.log_dir / "traces.log"

    def export(self, spans):
        with open(self.trace_file, "a") as f:
            for span in spans:
                f.write(json.dumps(self._format_span(span)) + "\n")
        return True

    def shutdown(self):
        pass

    def _format_span(self, span):
        return {
            "name": span.name,
            "trace_id": format(span.context.trace_id, "x"),
            "span_id": format(span.context.span_id, "x"),
            "parent_id": format(span.parent.span_id, "x") if span.parent else None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "attributes": dict(span.attributes),
            "status": span.status.status_code.name,
        }


class FileMetricExporter(MetricExporter):
    """File metric exporter for OpenTelemetry."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.log"
        super().__init__()

    def export(self, metrics_data: MetricsData, **kwargs) -> bool:
        """Export metrics data to a file.

        Args:
            metrics_data: The metrics data to export.

        Returns:
            True if the export was successful, False otherwise.
        """
        try:
            with open(self.metrics_file, "a") as f:
                for resource_metrics in metrics_data.resource_metrics:
                    for scope_metrics in resource_metrics.scope_metrics:
                        for metric in scope_metrics.metrics:
                            f.write(json.dumps(self._format_metric(metric)) + "\n")
            return True
        except Exception:
            return False

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> bool:
        """Shuts down the exporter.

        Args:
            timeout_millis: Time to wait for the export to complete in milliseconds.

        Returns:
            True if the shutdown succeeded, False otherwise.
        """
        return True

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Force flush the exporter.

        Args:
            timeout_millis: Time to wait for the flush to complete in milliseconds.

        Returns:
            True if the flush succeeded, False otherwise.
        """
        return True

    def _preferred_temporality(self) -> Dict:
        """Returns the preferred temporality for each instrument kind."""
        return {
            "counter": AggregationTemporality.CUMULATIVE,
            "up_down_counter": AggregationTemporality.CUMULATIVE,
            "observable_counter": AggregationTemporality.CUMULATIVE,
            "observable_up_down_counter": AggregationTemporality.CUMULATIVE,
            "histogram": AggregationTemporality.CUMULATIVE,
            "observable_gauge": AggregationTemporality.CUMULATIVE,
        }

    def _format_metric(self, metric):
        return {
            "name": metric.name,
            "description": metric.description,
            "unit": metric.unit,
            "data": self._format_data(metric.data),
        }

    def _format_data(self, data):
        if hasattr(data, "data_points"):
            return {
                "data_points": [
                    {
                        "attributes": dict(point.attributes),
                        "value": point.value if hasattr(point, "value") else None,
                        "count": point.count if hasattr(point, "count") else None,
                        "sum": point.sum if hasattr(point, "sum") else None,
                        "timestamp": point.time_unix_nano,
                    }
                    for point in data.data_points
                ]
            }
        return {}


class RetryingOTLPMetricExporter(MetricExporter):
    """A wrapper around OTLPMetricExporter that adds better retry logic."""
    
    def __init__(self, endpoint, headers=None, timeout=10):
        self.exporter = OTLPMetricExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout
        )
        self.max_retries = OTLP_MAX_RETRIES
        self.retry_delay = OTLP_RETRY_DELAY
        self.logger = logging.getLogger(__name__)
        super().__init__()
    
    def export(self, metrics_data, **kwargs):
        """Export metrics with retry logic for handling connection issues."""
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                return self.exporter.export(metrics_data, **kwargs)
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    ProtocolError, 
                    ReadTimeoutError) as e:
                last_exception = e
                retries += 1
                
                if retries <= self.max_retries:
                    # Use exponential backoff
                    delay = self.retry_delay * (2 ** (retries - 1))
                    # self.logger.warning(
                    #     f"Honeycomb export attempt {retries} failed: {str(e)}. "
                    #     f"Retrying in {delay}s..."
                    # )
                    time.sleep(delay)
                # else:
                    # self.logger.error(
                    #     f"Failed to export to Honeycomb after {retries} attempts: {str(e)}"
                    # )
            except Exception as e:
                # For non-connection errors, don't retry
                # self.logger.error(f"Unexpected error exporting to Honeycomb: {str(e)}")
                return False
                
        # If we get here, all retries failed
        return False
        
    def shutdown(self, timeout_millis=30000, **kwargs):
        """Shutdown the exporter."""
        return self.exporter.shutdown(timeout_millis, **kwargs)
        
    def force_flush(self, timeout_millis=10000):
        """Force flush the exporter."""
        return self.exporter.force_flush(timeout_millis)
        
    def _preferred_temporality(self):
        """Returns the preferred temporality."""
        return self.exporter._preferred_temporality()


class RetryingOTLPSpanExporter:
    """A wrapper around OTLPSpanExporter that adds better retry logic."""
    
    def __init__(self, endpoint, headers=None, timeout=10):
        self.exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout
        )
        self.max_retries = OTLP_MAX_RETRIES
        self.retry_delay = OTLP_RETRY_DELAY
        self.logger = logging.getLogger(__name__)
    
    def export(self, spans):
        """Export spans with retry logic for handling connection issues."""
        retries = 0
        
        while retries <= self.max_retries:
            try:
                return self.exporter.export(spans)
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    ProtocolError, 
                    ReadTimeoutError) as e:
                retries += 1
                
                if retries <= self.max_retries:
                    # Use exponential backoff
                    delay = self.retry_delay * (2 ** (retries - 1))
                    self.logger.warning(
                        f"Honeycomb trace export attempt {retries} failed: {str(e)}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Failed to export traces to Honeycomb after {retries} attempts: {str(e)}"
                    )
            except Exception as e:
                # For non-connection errors, don't retry
                self.logger.error(f"Unexpected error exporting traces to Honeycomb: {str(e)}")
                return False
                
        # If we get here, all retries failed
        return False
        
    def shutdown(self):
        """Shutdown the exporter."""
        return self.exporter.shutdown()
        
    def force_flush(self):
        """Force flush the exporter."""
        try:
            return self.exporter.force_flush()
        except Exception as e:
            self.logger.error(f"Error during trace force_flush: {str(e)}")
            return False


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
        if not TELEMETRY_ENABLED:
            return

        self._usage_records: List[UsageRecord] = []
        self._user_totals = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()
        self._installation_id = get_installation_id()

        # Initialize OpenTelemetry with more detailed resource attributes
        resource = Resource.create({
            "service.name": SERVICE_NAME,
            "service.version": os.getenv("DATABRIDGE_VERSION", "unknown"),
            "installation.id": self._installation_id,
            "environment": os.getenv("ENVIRONMENT", "production"),
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.0.0"
        })

        # Initialize tracing with both file and OTLP exporters
        tracer_provider = TracerProvider(resource=resource)
        
        # Always use both exporters
        log_dir = Path("logs/telemetry")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file exporter for local logging
        file_span_processor = BatchSpanProcessor(FileSpanExporter(str(log_dir)))
        tracer_provider.add_span_processor(file_span_processor)
        
        # Add Honeycomb OTLP exporter with retry logic
        if HONEYCOMB_ENABLED:
            # Create BatchSpanProcessor with improved configuration
            otlp_span_processor = BatchSpanProcessor(
                RetryingOTLPSpanExporter(
                    endpoint=OTLP_TRACES_ENDPOINT,
                    headers=OTLP_HEADERS,
                    timeout=OTLP_TIMEOUT,
                ),
                # Configure batch processing settings
                max_queue_size=OTLP_MAX_QUEUE_SIZE,
                max_export_batch_size=OTLP_MAX_EXPORT_BATCH_SIZE,
                schedule_delay_millis=OTLP_SCHEDULE_DELAY_MILLIS,
            )
            tracer_provider.add_span_processor(otlp_span_processor)

        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        # Initialize metrics with both exporters
        metric_readers = [
            # Local file metrics reader
            PeriodicExportingMetricReader(
                FileMetricExporter(str(log_dir)),
                export_interval_millis=60000,  # Export every minute
            ),
        ]

        # Add Honeycomb metrics reader if API key is available
        if HONEYCOMB_ENABLED:
            try:
                # Configure the OTLP metric exporter with improved error handling
                otlp_metric_exporter = RetryingOTLPMetricExporter(
                    endpoint=OTLP_METRICS_ENDPOINT,
                    headers=OTLP_HEADERS,
                    timeout=OTLP_TIMEOUT,
                )
                
                # Configure the metrics reader with improved settings
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        otlp_metric_exporter,
                        export_interval_millis=OTLP_SCHEDULE_DELAY_MILLIS,
                        export_timeout_millis=OTLP_TIMEOUT * 1000,
                    )
                )
                print(f"Successfully configured Honeycomb metrics exporter to {OTLP_METRICS_ENDPOINT}")
            except Exception as e:
                print(f"Failed to configure Honeycomb metrics exporter: {str(e)}")

        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)
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

    @asynccontextmanager
    async def track_operation(
        self,
        operation_type: str,
        user_id: str,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking operations with both usage metrics and OpenTelemetry.
        The user_id is hashed to ensure anonymity.
        """
        if not TELEMETRY_ENABLED:
            yield None
            return

        start_time = time.time()
        status = "success"
        current_span = trace.get_current_span()

        # Hash the user ID for anonymity
        hashed_user_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        try:
            # Add operation attributes to the current span
            current_span.set_attribute("operation.type", operation_type)
            current_span.set_attribute("user.id", hashed_user_id)
            if metadata:
                # Create a copy of metadata to avoid modifying the original
                metadata_copy = metadata.copy()
                
                # Remove the nested 'metadata' field completely if it exists
                if 'metadata' in metadata_copy:
                    del metadata_copy['metadata']
                
                # Set attributes for all remaining metadata fields
                for key, value in metadata_copy.items():
                    current_span.set_attribute(f"metadata.{key}", str(value))

            yield current_span

        except Exception as e:
            status = "error"
            current_span.set_status(Status(StatusCode.ERROR))
            current_span.record_exception(e)
            raise
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Record metrics
            attributes = {
                "operation": operation_type,
                "status": status,
                "installation_id": self._installation_id
            }
            self.operation_counter.add(1, attributes)
            if tokens_used > 0:
                self.token_counter.add(tokens_used, attributes)
            self.operation_duration.record(duration, attributes)

            # Record usage
            # Create a sanitized copy of metadata for the usage record
            sanitized_metadata = None
            if metadata:
                sanitized_metadata = metadata.copy()
                # Remove the nested 'metadata' field completely if it exists
                if 'metadata' in sanitized_metadata:
                    del sanitized_metadata['metadata']
            
            record = UsageRecord(
                timestamp=datetime.now(),
                operation_type=operation_type,
                tokens_used=tokens_used,
                user_id=hashed_user_id,
                duration_ms=duration,
                status=status,
                metadata=sanitized_metadata,
            )

            with self._lock:
                self._usage_records.append(record)
                self._user_totals[hashed_user_id][operation_type] += tokens_used

    def get_user_usage(self, user_id: str) -> Dict[str, int]:
        """Get usage statistics for a user."""
        if not TELEMETRY_ENABLED:
            return {}
            
        hashed_user_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        with self._lock:
            return dict(self._user_totals[hashed_user_id])

    def get_recent_usage(
        self,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Get recent usage records with optional filtering."""
        if not TELEMETRY_ENABLED:
            return []
            
        with self._lock:
            records = self._usage_records.copy()

        # Apply filters
        if user_id:
            hashed_user_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            records = [r for r in records if r.user_id == hashed_user_id]
        if operation_type:
            records = [r for r in records if r.operation_type == operation_type]
        if since:
            records = [r for r in records if r.timestamp >= since]
        if status:
            records = [r for r in records if r.status == status]

        return records
