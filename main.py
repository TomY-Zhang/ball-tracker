# flake8: noqa
from coco_model import CocoModel
from detector import Detector
from tracker import Tracker

MODEL_URLS = [
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz",
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
]
VIDEO_PATHS = [
    ".media/ball.mp4",
    ".media/objectTracking_examples_multiObject.avi"
]
CACHE_DIR = ".pretrained_models"
THRESHOLD = 0.2

def main():
    try:
        cm = CocoModel(model_url=MODEL_URLS[0], cache_dir=CACHE_DIR)

        detector = Detector(cm.model)
        detector.detect_video(VIDEO_PATHS[1], threshold=THRESHOLD)

        tracker = Tracker(cm.model)
        tracker.track_video(VIDEO_PATHS[1], threshold=THRESHOLD)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt: process aborted")
    except Exception as e:
        print("\n" + str(e))

    print("Exiting...")

if __name__ == '__main__':
    main()