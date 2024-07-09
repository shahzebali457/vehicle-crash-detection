# Derq Final Task

#### Problem / Objective 

- Develop a method to detect vehicle crashes in video clips by analyzing vehicle trajectories. 

## For Detector we are using Ultralytics Yolov10

## For tracking vehicle we are using open source UCMCTrack

## 🗼 Pipeline of UCMCTrack
First, the detection boxes are mapped onto the ground plane using homography transformation. Subsequently, the Correlated Measurement Distribution (CMD) of the target is computed. This distribution is then fed into a Kalman filter equipped with the Constant Velocity (CV) motion model and Process Noise Compensation (PNC). Next, the mapped measurement and the predicted track state are utilized as inputs to compute the Mapped Mahalanobis Distance (MMD). Finally, the Hungarian algorithm is applied to associate the mapped measurements with tracklets, thereby obtaining complete tracklets.
Link https://github.com/corfyi/UCMCTrack 




#### Environment
Before you begin, ensure you have the following prerequisites installed on your system:
- Python (3.8 or later)
- PyTorch with CUDA support
- Ultralytics Library
- Download weight file [yolov10m.pt](https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt to folder `pretrained`
- The current task is working good with yolov10m.pt. however you may use other variants too.
#### Run the demo

```bash
python demo_vid_area.py --video Input_videos/vid_4.mp4
python demo_vid_area.py --video Input_videos/vid_6.mp4
python demo_vid_area.py --video Input_videos/vid_14.mp4
```
-The output video, along with the detection, tracking, and crash JSON files, will be saved in the output/ folder



## 💁 Get Started
- Install the required dependency packages 

```bash
pip install -r requirements.txt
```

## To install Yolov10
- git clone https://github.com/THU-MIG/yolov10.git
- cd yolov10
- pip install .

## OPENmp Issue
- If you faced openmp lib issue in windows RUN the command
```bash
 set KMP_DUPLICATE_LIB_OK=TRUE
```





