# Ultralytics YOLO 🚀, AGPL-3.0 license
import sys 
sys.path.insert(0,'/home/altex/Codes/ultralytics')

from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel,DeTrackModel

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'detrack': {
                'model': DeTrackModel,
                'trainer': yolo.detect.DeTrackTrainer,
                'validator': yolo.detect.DeTrackValidator,
                'predictor': yolo.detect.DetTrackPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }


if __name__=='__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-track.yaml')
    import torch 
    x = torch.rand((1,3,608,608))
    out = model(x)
    print(out)

