import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLOv10
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import os 
import json
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
        self.model = YOLOv10('pretrained/yolov10s.pt')###################################################################

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
    
def save_json(data, filename):
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=convert)


def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)
# Function to calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    # print("IoU",intersection_area / union_area)
    return intersection_area / union_area
def calculate_distance(centroid1, centroid2):
    """
    Calculate the Euclidean distance between two detections based on their centroids.
    """
    return ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) ** 0.5


# def calculate_distance(det1, det2):
#     """
#     Calculate the Euclidean distance between two detections based on their centroids.
#     """
#     centroid1_x = det1.bb_left + det1.bb_width / 2
#     centroid1_y = det1.bb_top + det1.bb_height / 2
#     centroid2_x = det2.bb_left + det2.bb_width / 2
#     centroid2_y = det2.bb_top + det2.bb_height / 2

#     distance = math.sqrt((centroid1_x - centroid2_x) ** 2 + (centroid1_y - centroid2_y) ** 2)
#     return distance

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon
    """
    result = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (point[0], point[1]), False)
    return result >= 0

def main(args):
    class_list = [2, 3, 5]

    cap = cv2.VideoCapture(args.video)

    # Get the fps of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    output_video_path = os.path.join(output_folder, f'output_{Path(args.video).stem}.mp4')
    detection_output_path = os.path.join(output_folder, f'detections_{Path(args.video).stem}.json')
    tracking_output_path = os.path.join(output_folder, f'tracking_{Path(args.video).stem}.json')
    crash_output_path = os.path.join(output_folder, f'crashes_{Path(args.video).stem}.json')

    video_out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    #Define the polygon area (for example, a trapezoid in the middle of the frame)

    # Define scaling factors for width and height
    initial_width_factor = 0.1  
    final_width_factor = 0.95
    # initial_height_factor = 0.18 #0.4 , 0.18 , 0.17
    # final_height_factor = 0.4
    config = load_config()
    video_name = Path(args.video).stem
    vid_frame_height = config["vid_frame_height"]
    if video_name in vid_frame_height:
        initial_height_factor = vid_frame_height[video_name]
    else:
        initial_height_factor = 0.4


    # Define the custom polygon area using scaling factors
    polygon = [
        (int(width * initial_width_factor), int(height * initial_height_factor)), 
        (int(width * final_width_factor), int(height * initial_height_factor)), 
        (int(width * final_width_factor), height), 
        (int(width * 0.05), height)
    ]
    # Open a cv window and specify the height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, None)
    # Initialize previous_centroids dictionary to store previous centroids of detections
    previous_centroids = {} 
    collision_display_counter = 0 
    # Loop to read video frames
    frame_id = 1
    collision_counter = {}
    crash = False
    detections_output = []
    tracking_output = []
    crash_output = []
    while True:
        ret, frame_img = cap.read()
        if not ret:
            break

        
        dets = detector.get_dets(frame_img, args.conf_thresh, class_list)
        # Output detection and tracking results to JSON files
        
        
        tracker.update(dets, frame_id)
        for det in dets:
            detections_output.append({
                'frame': frame_id,
                'det_class': int(det.det_class),
                'bbox': [int(det.bb_left),int(det.bb_top), int(det.bb_width), int(det.bb_height)],
                'conf': float(det.conf),
                
            })
            if det.track_id > 0:        
                tracking_output.append({
                    'track_id': int(det.track_id),
                    'det_class': int(det.det_class),
                    'bbox': [int(det.bb_left),int(det.bb_top), int(det.bb_width), int(det.bb_height)],
                    'centroid': (int(det.bb_left + det.bb_width / 2), int(det.bb_top + det.bb_height / 2))
                    })
                
        # Measure and display distances between detected objects within the polygon
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                if dets[i].track_id > 0 and dets[j].track_id > 0:
                    centroid1_x = int(dets[i].bb_left + dets[i].bb_width / 2)
                    centroid1_y = int(dets[i].bb_top + dets[i].bb_height / 2)
                    centroid2_x = int(dets[j].bb_left + dets[j].bb_width / 2)
                    centroid2_y = int(dets[j].bb_top + dets[j].bb_height / 2)

                    current_centroid1 = (centroid1_x, centroid1_y)
                    current_centroid2 = (centroid2_x, centroid2_y)

                    if crash:
                        collision_display_counter -= 1
                        # Draw the combined bounding box in red
                        cv2.putText(frame_img, "Vehicle Crashed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.rectangle(frame_img, (int(dets[i].bb_left), int(dets[i].bb_top)), (int(dets[i].bb_left + dets[i].bb_width), int(dets[i].bb_top + dets[i].bb_height)), (0, 255, 0), 2)
                        cv2.rectangle(frame_img, (int(dets[j].bb_left), int(dets[j].bb_top)), (int(dets[j].bb_left + dets[j].bb_width), int(dets[j].bb_top + dets[j].bb_height)), (0, 255, 0), 2)

                        cv2.rectangle(frame_img, (combined_box_left, combined_box_top), (combined_box_right, combined_box_bottom), (0, 0, 255), 4)
                        crash_location = (
                            int((dets[i].bb_left + dets[j].bb_left + dets[i].bb_width) / 2),
                            int((dets[i].bb_top + dets[j].bb_top + dets[i].bb_height) / 2)
                        )
                        crash_output.append({
                            'id1': int(dets[i].track_id),
                            'id2': int(dets[j].track_id),
                            'location': crash_location})
                        if collision_display_counter == 0:
                            collision_display_counter  = 0
                            crash = False
                    else:

                        cv2.rectangle(frame_img, (int(dets[i].bb_left), int(dets[i].bb_top)), (int(dets[i].bb_left + dets[i].bb_width), int(dets[i].bb_top + dets[i].bb_height)), (0, 255, 0), 2)
                        cv2.rectangle(frame_img, (int(dets[j].bb_left), int(dets[j].bb_top)), (int(dets[j].bb_left + dets[j].bb_width), int(dets[j].bb_top + dets[j].bb_height)), (0, 255, 0), 2)
                        cv2.circle(frame_img, (centroid1_x, centroid1_y), 5, (235, 219, 11), -1)
                        cv2.circle(frame_img, (centroid2_x, centroid2_y), 5, (235, 219, 11), -1)
                        cv2.putText(frame_img, str(dets[i].track_id), (int(dets[i].bb_left), int(dets[i].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame_img, str(dets[j].track_id), (int(dets[j].bb_left), int(dets[j].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if point_in_polygon(current_centroid1, polygon) and point_in_polygon(current_centroid2, polygon):
                        current_distance = calculate_distance(current_centroid1, current_centroid2)

                        # Retrieve previous centroids if available
                        previous_centroid1 = previous_centroids.get(dets[i].track_id)
                        previous_centroid2 = previous_centroids.get(dets[j].track_id)

                        if previous_centroid1 is not None and previous_centroid2 is not None:
                            previous_distance = calculate_distance(previous_centroid1, previous_centroid2)

                            # Check if the objects are moving closer
                            if current_distance < previous_distance:
                                print(f"Distance between detection {dets[i].id} and detection {dets[j].id}: {current_distance:.2f}")

                                cv2.line(frame_img, (centroid1_x, centroid1_y), (centroid2_x, centroid2_y), (255, 0, 0), 2)
                                cv2.putText(frame_img, f"{current_distance:.2f}", ((centroid1_x + centroid2_x) // 2, (centroid1_y + centroid2_y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                                # Check for bounding box overlap
                                box1 = [int(dets[i].bb_left), int(dets[i].bb_top), int(dets[i].bb_left + dets[i].bb_width), int(dets[i].bb_top + dets[i].bb_height)]
                                box2 = [int(dets[j].bb_left), int(dets[j].bb_top), int(dets[j].bb_left + dets[j].bb_width), int(dets[j].bb_top + dets[j].bb_height)]
                                iou = calculate_iou(box1, box2)
                                print("IoU", iou)

                                if 0.03 <= iou <= 0.5 and int(current_distance) <= 75:
                                    print(f"Accident detected between detection {dets[i].id} and detection {dets[j].id}")

                                    pair_key = (dets[i].track_id, dets[j].track_id)

                                    # Increment the collision counter for this pair
                                    if pair_key in collision_counter:
                                        collision_counter[pair_key] += 1
                                    else:
                                        collision_counter[pair_key] = 1
                                    # Create a combined bounding box
                                    combined_box_left = min(box1[0], box2[0])
                                    combined_box_top = min(box1[1], box2[1])
                                    combined_box_right = max(box1[2], box2[2])
                                    combined_box_bottom = max(box1[3], box2[3])

                                    if not crash:
                                        if collision_counter[pair_key] >= 3:
                                            collision_display_counter = 150
                                            crash = True
                                            # cv2.putText(frame_img, "Collision", (combined_box_left, combined_box_top - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                                        else:
                                            crash = False
                                    
                                else:
                                    pair_key = (dets[i].track_id, dets[j].track_id)
                                    collision_counter[pair_key] = 0 
        # Update previous centroids at the end of the frame processing
        for det in dets:
            if det.track_id > 0:
                centroid_x = int(det.bb_left + det.bb_width / 2)
                centroid_y = int(det.bb_top + det.bb_height / 2)
                previous_centroids[det.track_id] = (centroid_x, centroid_y)

        frame_id += 1
        # Draw the transparent polygon area
        overlay = frame_img.copy()
        cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], (0, 255, 0))
        alpha = 0.4  # Transparency factor
        frame_img = cv2.addWeighted(overlay, alpha, frame_img, 1 - alpha, 0)

        # Display the current frame
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)

        video_out.write(frame_img)

    save_json(detections_output, detection_output_path)
    save_json(tracking_output, tracking_output_path)
    save_json(crash_output, crash_output_path)
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default="Input_videos/vid_6.mp4", help='video file name')
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
