# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, DetTrackPredictor
from .train import DetectionTrainer, DeTrackTrainer
from .val import DetectionValidator,DeTrackValidator

__all__ = 'DetectionPredictor', 'DetectionTrainer', 'DetectionValidator', 'DetTrackPredictor', 'DeTrackValidator','DeTrackTrainer'
