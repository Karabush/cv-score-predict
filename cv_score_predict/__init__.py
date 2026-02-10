from .core import cv_score_predict

__all__ = ["cv_score_predict"]

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"  # During development before install