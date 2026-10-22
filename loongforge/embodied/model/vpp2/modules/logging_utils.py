"""Small logging adapter used by the transplanted VPP2 modules."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a standard-library logger without importing BASE utilities."""
    return logging.getLogger(name)
