from ultralytics import YOLO
import os, cv2
import argparse
from pathlib import Path
from ultralytics import YOLOv10
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np
import math

# Define a Detection class containing id, bb_left, bb_top, bb_width, bb_height, conf, det_class
class Detection:

    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left + self.bb_width / 2, self.bb_top + self.bb_height, self.y[0, 0], self.y[1, 0])

    def __repr__(self):
        return self.__str__()


# Detector class, used to get target detection results from Yolo detector
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    
    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
        # self.model = YOLO('pretrained/yolov8n.pt')
        self.model = YOLOv10('pretrained/yolov10n.pt')

    def get_dets(self, img, conf_thresh=0, det_classes=[0]):

        dets = []

        # Convert the frame from BGR to RGB (because OpenCV uses BGR format)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use RTDETR for inference
        results = self.model(frame, imgsz=1088)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # Create a new Detection object
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1

            dets.append(det)

        return dets

def calculate_distance(det1, det2):
    """
    Calculate the Euclidean distance between two detections based on their centroids.
    """

    centroid1_x = det1.bb_left + det1.bb_width / 2
    centroid1_y = det1.bb_top + det1.bb_height / 2
    centroid2_x = det2.bb_left + det2.bb_width / 2
    centroid2_y = det2.bb_top + det2.bb_height / 2

    distance = math.sqrt((centroid1_x - centroid2_x) ** 2 + (centroid1_y - centroid2_y) ** 2)
    return distance

def main(args):

    class_list = [2, 5, 7]

    cap = cv2.VideoCapture(args.video)

    # Get the fps of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    output_video_path = os.path.join(output_folder, f'output_{Path(args.video).stem}.mp4')
    video_out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Open a cv window and specify the height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, None)

    # Loop to read video frames
    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret:
            break

        dets = detector.get_dets(frame_img, args.conf_thresh, class_list)
        tracker.update(dets, frame_id)

        for det in dets:

            # Draw detection box
            if det.track_id > 0:
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)), (0, 255, 0), 2)
                # Draw the id of the detection box
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Calculate and draw the centroid
                centroid_x = int(det.bb_left + det.bb_width / 2)
                centroid_y = int(det.bb_top + det.bb_height / 2)

                cv2.circle(frame_img, (centroid_x, centroid_y), 5, (235, 219, 11), -1)
                # points = np.array([(x, y) for x, y, _ in track], dtype=np.int32).reshape((-1, 1, 2))
                # cv2.polylines(im, [points], isClosed=False, color=(37, 255, 225), thickness=2)
                # cv2.circle(im, (int(track[-1][0]), int(track[-1][1])), 5, (235, 219, 11), -1)


        # Measure and display distances between detected objects
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                if det.track_id > 0:
                    distance = calculate_distance(dets[i], dets[j])
                    print(f"Distance between detection {dets[i].id} and detection {dets[j].id}: {distance:.2f}")
                    centroid1_x = int(dets[i].bb_left + dets[i].bb_width / 2)
                    centroid1_y = int(dets[i].bb_top + dets[i].bb_height / 2)
                    centroid2_x = int(dets[j].bb_left + dets[j].bb_width / 2)
                    centroid2_y = int(dets[j].bb_top + dets[j].bb_height / 2)

                    cv2.line(frame_img, (centroid1_x, centroid1_y), (centroid2_x, centroid2_y), (255, 0, 0), 2)
                    cv2.putText(frame_img, f"{distance:.2f}", ((centroid1_x + centroid2_x) // 2, (centroid1_y + centroid2_y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_id += 1

        # Display the current frame
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)

        video_out.write(frame_img)

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default="Input_videos/vid_4.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default="demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
args = parser.parse_args()

main(args)
