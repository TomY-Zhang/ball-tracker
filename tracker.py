# flake8: noqa
import cv2, time
import tensorflow as tf
import numpy as np
from ball import Ball

SPORTS_BALL_ID = 37

class Tracker:
    def __init__(self, model):
        self.model = model
        self.balls = []

    # Find closest ball in 1d plane
    def find_closest_ball(self, x):
        idx = 0
        min_dist = float('inf')

        for i in range(len(self.balls)):
            pos = self.balls[i].get_pos()
            if abs(pos - x) < min_dist:
                idx = i
                min_dist = abs(pos - x)
                
        return idx

    def track_frame(self, image, threshold=0.5):
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)[tf.newaxis, ...]

        # feed image into pretrained model
        detections = self.model(input_tensor)
        bboxs = detections['detection_boxes'][0].numpy()

        # get class names and prediction scores from model
        class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detections['detection_scores'][0].numpy()

        # reduce noise by filtering out classes with prediction scores below the threshold
        above_thresh = tf.image.non_max_suppression(bboxs, class_scores, max_output_size=50,
                                                    iou_threshold=threshold, score_threshold=threshold)
        
        # filter out all objects that are not sports balls
        ball_detections = [i for i in above_thresh.numpy() if class_ids[i] == SPORTS_BALL_ID]
        
        # if # of balls detected > # of balls currently tracked, add more balls to track
        num_detect = len(ball_detections)
        while len(self.balls) < num_detect:
            self.balls.append(Ball())

        # draw bounding boxes on image
        imgH, imgW, _ = image.shape
        if len(bboxs) != 0:
            for i in ball_detections:
                bbox = tuple(bboxs[i].tolist())

                # unpack bouding box coords and scale to image dimensions
                ymin, xmin, ymax, xmax = bbox
                ymin, ymax = int(ymin * imgH), int(ymax * imgH)

                # get center x and y of boudning box
                cx = int((xmin + xmax) * imgW / 2)
                cy = int((ymin + ymax) / 2)

                # find nearest ball from detection and predict next position
                idx = self.find_closest_ball(cx)
                self.balls[idx].predict()
                self.balls[idx].update(cx)
                pos = int(self.balls[idx].get_pos())
                
                # draw a rectangle with the bbox coords
                cv2.circle(img=image, center=(pos, cy), radius=20, color=(0, 255, 0), thickness=2)

                # add label text for each class
                ball_confidence = round(100 * class_scores[i])
                display_text = f"Sports ball: {ball_confidence}"
                cv2.putText(image, display_text, (pos-50, ymin - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        return image
    
    # capture and track objects in video
    def track_video(self, video_path, threshold=0.5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error opening file '{video_path}'")

        filename = "ball_track.mp4"
        print(f"Generating '{filename}'...")

        fps = 5 # frames per second

        # get frame dimensions
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
    
        # Create video with trackers
        video = cv2.VideoWriter(f'./results/{filename}', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        (success, frame) = cap.read()
        while success:
            # detect and track balls on each frame of video
            image = self.track_frame(frame, threshold)
            video.write(image)

            (success, frame) = cap.read()
        
        cv2.destroyAllWindows()
        cap.release()
        video.release()

        print(f"'{filename}' successfully created!")