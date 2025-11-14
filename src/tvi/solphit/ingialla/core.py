from __future__ import annotations

from tvi.solphit.base import SolphitLogger

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

log = SolphitLogger.get_logger("tvi.solphit.ingialla")

def hello(name: str) -> str:
    """Tiny exemplar function to prove imports/logging work."""
    msg = f"Ingialla says hello to {name}"
    log.info(msg)
    return msg

def cosine_similarity(a, b):
    """Example function using numpy (if available)."""
    if np is None:
        raise RuntimeError("numpy not installed or failed to import.")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)