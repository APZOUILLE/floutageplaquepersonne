from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import ByteTrack

tracker = ByteTrack()

def update_tracks(results):
    detections = Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    return detections

