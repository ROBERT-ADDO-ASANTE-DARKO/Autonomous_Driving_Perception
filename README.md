# Autonomous Driving Perception System

This project implements an advanced perception system for autonomous driving, combining object detection, tracking, depth estimation, and visualization techniques. It integrates YOLOv9 for object detection, DeepSORT for multi-object tracking, and DPT for monocular depth estimation. Additionally, it provides pseudo-LiDAR BEV (Bird's Eye View) visualization and 3D bounding box stabilization.

## Features

- **Object Detection**: YOLOv9-based object detection for identifying vehicles, pedestrians, and other objects.
- **Multi-Object Tracking**: DeepSORT for tracking objects across frames.
- **Depth Estimation**: Monocular depth estimation using DPT models.
- **Pseudo-LiDAR Visualization**: BEV visualization of the scene using depth maps.
- **3D Bounding Boxes**: Stabilized 3D bounding box projection with Kalman filtering.
- **Safety Analysis**: Distance-based safety alerts for vehicles.
- **Point Cloud Export**: Export pseudo-LiDAR point clouds in `.ply` format.

## Project Structure

```
.
├── av_perception_system.py  # Main implementation file
├── depth_log.csv            # Log file for depth-related data
```

### Key Classes and Functions

- **[`DepthEstimator`](av_perception_system.py)**: Estimates depth maps from RGB images.
- **[`EnhancedPseudoLidarBEV`](av_perception_system.py)**: Generates BEV visualization from depth maps.
- **[`Stabilized3DBoundingBox`](av_perception_system.py)**: Stabilizes and projects 3D bounding boxes.
- **[`initialize_deepsort`](av_perception_system.py)**: Initializes the DeepSORT tracker.
- **[`draw_boxes_with_depth_improved`](av_perception_system.py)**: Draws bounding boxes with depth information.

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision transformers opencv-python numpy scipy pillow open3d
```

Additionally, ensure the following repositories are available:
- [YOLOv9](https://github.com/ultralytics/yolov5) (for object detection)
- [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch) (for object tracking)
- [YOLOP](https://github.com/hustvl/YOLOP) (optional, for lane and drivable area segmentation)

## Usage

### Running the System

To run the perception system, use the following command:

```bash
python av_perception_system.py --source <input_source> --weights <path_to_yolo_weights>
```

### Arguments

- `--source`: Input source (image, video, or webcam).
- `--weights`: Path to YOLOv9 weights.
- `--conf-thres`: Confidence threshold for object detection.
- `--iou-thres`: IoU threshold for non-max suppression.
- `--save-frames`: Save individual frames with detections.
- `--draw-trails`: Draw tracking trails for objects.

### Example

```bash
python av_perception_system.py --source data/video.mp4 --weights yolov9.pt --save-frames --draw-trails
```

## Outputs

- **Annotated Video**: The processed video with object detection, tracking, and depth overlays.
- **Point Clouds**: Exported `.ply` files for pseudo-LiDAR visualization.
- **Logs**: Depth-related data logged in `depth_log.csv`.

## Acknowledgments

This project leverages the following open-source libraries and models:
- [YOLOv9](https://github.com/ultralytics/yolov5)
- [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch)
- [DPT Depth Estimation](https://huggingface.co/Intel/dpt-large)
- [YOLOP](https://github.com/hustvl/YOLOP)

## License

This project is licensed under the MIT