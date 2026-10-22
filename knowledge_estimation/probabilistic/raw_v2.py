"""Conservative raw revision: duplicate events advance processing time.

No score mapping, new probabilities or sensor assumptions. Keep v1 intact so
old frozen artifacts can still be reproduced. Explicitly opt into this class.
"""

from functools import wraps
import hashlib
from pathlib import Path

from .estimator import KnowledgeEstimator


def _advance_clock(method):
    @wraps(method)
    def handle(self, *args, **kwargs):
        # Idempotency applies to evidence, not the passage of time. In
        # particular an unchanged ACK cannot freeze C's validity indefinitely.
        self._tick(kwargs.get("now"))
        return method(self, *args, **kwargs)
    return handle


class RawKnowledgeEstimatorV2(KnowledgeEstimator):
    """Same evidence model as v1, with monotonic time on replayed events.

    Invalid calls may advance the processing clock but never add evidence.
    Callers must use serialized monotonic timestamps, as required by v1.
    """

    record_send = _advance_clock(KnowledgeEstimator.record_send)
    observe_ack = _advance_clock(KnowledgeEstimator.observe_ack)
    observe_ack_timeout = _advance_clock(KnowledgeEstimator.observe_ack_timeout)
    record_result = _advance_clock(KnowledgeEstimator.record_result)
    verify_task = _advance_clock(KnowledgeEstimator.verify_task)

    @property
    def fingerprint(self):
        code = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(b"raw-v2:" + super().fingerprint.encode() + code).hexdigest()
