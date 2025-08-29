import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
from transformers import pipeline
import time
from PIL import Image

import sys
sys.path.append("../YOLOP")  # Add YOLOP to Python path

from lib.config import cfg
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
import torchvision.transforms as transforms
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import csv

# Setup log path
depth_log_path = Path("depth_log.csv")
if not depth_log_path.exists():
    with open(depth_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame_id", "object_id", "class_id", "label", "bbox_width_px",
            "raw_depth_m", "corrected_depth_m", "fx_estimated", "fx_nominal"
        ])

calibrated_pixels_per_meter = None  # will be set dynamically from first car

previous_depths = {}

'''
class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large"):
        """
        Initialize depth estimation model
        Options: "Intel/dpt-large", "Intel/dpt-hybrid-midas", "nielsr/dpt-large"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_estimator = pipeline(
            "depth-estimation", 
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def estimate_depth(self, image):
        """
        Estimate depth map from RGB image.
        Returns: 
        - raw depth map (in meters)
        - visualized depth (uint8 3-channel)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Run depth estimation
        depth_result = self.depth_estimator(pil_image)
        depth_map = np.array(depth_result["depth"])  # in meters

        # Normalize for visualization only
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

        return depth_map, depth_vis
    
    def get_object_distance(self, depth_map, bbox, camera_params=None):
        """
        Calculate average distance of object from bounding box region
        
        Args:
            depth_map: Raw depth map from model
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            camera_params: Optional camera calibration parameters
        
        Returns:
            distance_meters: Estimated distance in meters
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Extract depth values from bounding box region
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # Use median depth to avoid outliers
        median_depth = np.median(roi_depth)
        
        # Convert relative depth to real-world distance
        # This is a simplified conversion - in practice you'd need camera calibration
        # For now, we'll use an empirical scaling factor
        distance_meters = self.depth_to_distance(median_depth, camera_params)
        
        return distance_meters
    
    def depth_to_distance(self, depth_value, camera_params=None):
        """
        Convert depth map value to real-world distance
        This is simplified - real implementation would use camera intrinsics
        """
        # Empirical conversion for typical automotive cameras
        # You would calibrate this based on your specific camera setup
        max_distance = 100  # Maximum detection distance in meters
        min_distance = 1    # Minimum detection distance in meters
        
        # Invert and scale depth value
        normalized_depth = depth_value / np.max(depth_value) if hasattr(depth_value, '__len__') else depth_value
        distance = min_distance + (max_distance - min_distance) * (1 - normalized_depth)
        
        return max(distance, min_distance)
'''

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import torch
import numpy as np
import cv2

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def estimate_depth(self, image_bgr):
        """
        Estimate depth from an input BGR image.
        Returns:
        - depth_map: float32 (H, W) in meters
        - depth_vis: colorized uint8 image for display
        """
        import torch.nn.functional as F

        # Convert to RGB and PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Prepare inputs for model
        inputs = self.feature_extractor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth_pred = outputs.predicted_depth  # shape: [1, H, W]
            if depth_pred.ndim == 3:
                depth_pred = depth_pred.unsqueeze(1)  # → [1, 1, H, W]

        # Resize to match input image size (H, W)
        target_size = image_bgr.shape[:2]  # (H, W)
        depth_resized = F.interpolate(
            depth_pred,
            size=target_size,
            mode="bicubic",
            align_corners=False
        )

        # Remove batch and channel dims → shape: (H, W)
        depth_map = depth_resized.squeeze().cpu().numpy().astype(np.float32)

        # Colorize for visualization
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

        return depth_map, depth_vis

import numpy as np
import cv2

class YawKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(2, 1)  # [yaw, delta_yaw]
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.array([[1e-3, 0], [0, 1e-4]], dtype=np.float32)
        self.kf.measurementNoiseCov = np.array([[1e-2]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 0.1
        self.kf.statePost = np.zeros((2, 1), dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0])

    def correct(self, measured_yaw):
        # Wrap angle to [-pi, pi]
        yaw = (measured_yaw + np.pi) % (2 * np.pi) - np.pi
        corrected = self.kf.correct(np.array([[yaw]], dtype=np.float32))
        return float(corrected[0])

class PositionKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # [x, y, dx, dy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def correct(self, measured_x, measured_y):
        meas = np.array([[measured_x], [measured_y]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        return int(corrected[0]), int(corrected[1])

from collections import deque
import numpy as np
import cv2
from scipy.spatial import cKDTree  # Faster implementation
from collections import defaultdict

class EnhancedPseudoLidarBEV:
    def __init__(self, canvas_size=400, scale=5.0, buffer_size=15, max_range=80.0):
        self.canvas_size = canvas_size
        self.scale = scale
        self.buffer = deque(maxlen=buffer_size)
        self.max_range = max_range
        self.center_x = canvas_size // 2
        self.center_y = canvas_size - 20
        self.height_colormap = cv2.COLORMAP_JET
        self.grid_spacing = 10
        self.grid_color = (50, 50, 50)
        self.vehicle_history = defaultdict(lambda: deque(maxlen=20))
        self.clean_frame = None  # will be set externally before update_and_overlay

    def depth_to_bev_points(self, depth_map, K, clean_rgb_frame=None):
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        mask = (ys > h * 0.4) & (depth_map > 0.1) & (depth_map < self.max_range)

        z = depth_map[mask]
        x = (xs[mask] - cx) * z / fx
        y = (ys[mask] - cy) * z / fy

        if clean_rgb_frame is not None:
            colors = clean_rgb_frame[ys[mask], xs[mask]]
        else:
            norm_height = np.clip((y + 2.0) / 4.0, 0, 1)
            colors = (norm_height * 255).astype(np.uint8)
            colors = cv2.applyColorMap(colors, self.height_colormap)
            colors = colors.squeeze()

        return np.column_stack((x, y, z)), colors

    def render_bev(self, points, colors=None):
        """
        Render a top-down BEV canvas from 2D or 3D point input.
        If points include y (height), it's ignored in rendering.
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self._draw_grid(canvas)

        if len(points) == 0:
            return canvas

        # If 3D points: extract x and z only
        if points.shape[1] == 3:
            x, z = points[:, 0], points[:, 2]
        elif points.shape[1] == 2:
            x, z = points[:, 0], points[:, 1]
        else:
            raise ValueError("Expected point shape (N, 2) or (N, 3), got: " + str(points.shape))

        canvas_x = (self.center_x + x * self.scale).astype(int)
        canvas_y = (self.center_y - z * self.scale).astype(int)

        valid = (canvas_x >= 0) & (canvas_x < self.canvas_size) & \
                (canvas_y >= 0) & (canvas_y < self.canvas_size)

        canvas_x = canvas_x[valid]
        canvas_y = canvas_y[valid]

        if colors is not None:
            colors = colors[valid]
            for x_px, y_px, color in zip(canvas_x, canvas_y, colors):
                cv2.circle(canvas, (x_px, y_px), 1, color.tolist(), -1)
        else:
            for x_px, y_px in zip(canvas_x, canvas_y):
                cv2.circle(canvas, (x_px, y_px), 1, (0, 255, 255), -1)

        return canvas

    def _draw_grid(self, canvas):
        for x in range(-50, 51, self.grid_spacing):
            canvas_x = int(self.center_x + x * self.scale)
            if 0 <= canvas_x < self.canvas_size:
                cv2.line(canvas, (canvas_x, 0), (canvas_x, self.canvas_size), self.grid_color, 1)
                cv2.putText(canvas, f"{abs(x)}m", (canvas_x - 15, self.canvas_size - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for z in range(0, int(self.max_range) + 1, self.grid_spacing):
            canvas_y = int(self.center_y - z * self.scale)
            if 0 <= canvas_y < self.canvas_size:
                cv2.line(canvas, (0, canvas_y), (self.canvas_size, canvas_y), self.grid_color, 1)
                cv2.putText(canvas, f"{z}m", (10, canvas_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def update_vehicle_positions(self, tracked_boxes, categories, depth_map, K):
        for i, box in enumerate(tracked_boxes):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = y2
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                z = depth_map[cy, cx]
                if 0.1 < z < self.max_range:
                    fx = K[0, 0]
                    cx0 = K[0, 2]
                    x = (cx - cx0) * z / fx
                    class_id = categories[i]
                    color = colorLabels(class_id) if 'colorLabels' in globals() else (0, 0, 255)
                    track_id = i  # ideally replace with real track ID
                    self.vehicle_history[track_id].append((x, z, color))

    def draw_vehicle_trails(self, canvas):
        for track_id, history in self.vehicle_history.items():
            if len(history) < 2:
                continue
            points = []
            for x, z, color in history:
                cx = int(self.center_x + x * self.scale)
                cy = int(self.center_y - z * self.scale)
                points.append((cx, cy))
                cv2.circle(canvas, (cx, cy), 3, color, -1)
            for i in range(1, len(points)):
                alpha = i / len(points)
                faded = tuple(int(c * alpha) for c in history[i][2])
                cv2.line(canvas, points[i - 1], points[i], faded, 1)

    def update_and_overlay(self, frame, depth_map, K, tracked_boxes=None, 
                           categories=None, overlay_pos=(10, 10)):
        if self.clean_frame is not None:
            points, colors = self.depth_to_bev_points(depth_map, K, self.clean_frame)
        else:
            points, colors = self.depth_to_bev_points(depth_map, K)

        self.buffer.append((points, colors))

        all_points = np.vstack([p for p, _ in self.buffer]) if self.buffer else np.array([])
        all_colors = np.vstack([c for _, c in self.buffer]) if self.buffer else None
        bev_map = self.render_bev(all_points, all_colors)

        # === Draw bounding boxes in BEV
        if tracked_boxes is not None and categories is not None:
            for i, box in enumerate(tracked_boxes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = y2
                if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                    z = depth_map[cy, cx]
                    if 0.1 < z < self.max_range:
                        fx = K[0, 0]
                        cx0 = K[0, 2]
                        x = (cx - cx0) * z / fx
                        canvas_x = int(self.center_x + x * self.scale)
                        canvas_y = int(self.center_y - z * self.scale)
                        box_px = int(1.5 * self.scale)
                        color = colorLabels(categories[i]) if 'colorLabels' in globals() else (0, 0, 255)
                        cv2.rectangle(bev_map,
                                      (canvas_x - box_px, canvas_y - box_px),
                                      (canvas_x + box_px, canvas_y + box_px),
                                      color, 1)

            # Optional: Draw trails
            self.update_vehicle_positions(tracked_boxes, categories, depth_map, K)
            #self.draw_vehicle_trails(bev_map)

        #self._add_bev_legend(bev_map)

        h_bev, w_bev = bev_map.shape[:2]
        h_frame, w_frame = frame.shape[:2]
        x_offset = 10
        y_offset = h_frame - h_bev - 10
        if y_offset < 0:
            y_offset = 0
            scale_factor = h_frame / h_bev
            bev_map = cv2.resize(bev_map, (int(w_bev * scale_factor), h_frame))
            h_bev, w_bev = bev_map.shape[:2]

        try:
            frame[y_offset:y_offset + h_bev, x_offset:x_offset + w_bev] = bev_map
            cv2.putText(frame, "BEV Point Cloud", (x_offset, y_offset - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Warning: BEV overlay failed - {str(e)}")

        return frame

    def export_to_ply(self, filepath="bev_cloud.ply"):
        """
        Export the accumulated pseudo-lidar BEV point cloud as a .ply file.
        Points are in (x, 0, z) format with RGB color.
        """
        if not self.buffer:
            print("No points to export.")
            return

        all_points = np.vstack([p for p, _ in self.buffer])
        all_colors = np.vstack([c for _, c in self.buffer])
        
        # Format to (x, y=0, z) and RGB
        points_xyz = np.column_stack((all_points[:, 0], np.zeros_like(all_points[:, 0]), all_points[:, 1]))
        colors_rgb = all_colors[:, :3]

        # Write header
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

        print(f"✅ PLY point cloud saved to {filepath}")

    def export_full_pointcloud_to_ply(self, filepath="full_cloud.ply", K=None):
        """
        Export full 3D pseudo-lidar point cloud (x, y, z, r, g, b) to a .ply file.
        Uses most recent RGBD buffer and reprojects height (y).
        """
        if not self.buffer:
            print("⚠️ No data in buffer to export.")
            return

        all_points = []
        all_colors = []

        for (xz, colors) in self.buffer:
            if xz.shape[1] == 2:
                print("⚠️ Only XZ data available; can't export full 3D without Y.")
                return

        print("⚠️ This method assumes full (x, y, z) input; please modify depth_to_bev_points to return y.")

        # --- OPTIONAL: If you've already computed (x, y, z), change buffer structure and use this block:
        # for (xyz, colors) in self.buffer:
        #     all_points.append(xyz)
        #     all_colors.append(colors)
        #
        # all_points = np.vstack(all_points)
        # all_colors = np.vstack(all_colors)

        # --- Example fallback to 2.5D using zeros for Y-axis:
        for (xz, colors) in self.buffer:
            xz = np.asarray(xz)
            xyz = np.column_stack((xz[:, 0], np.zeros_like(xz[:, 0]), xz[:, 1]))
            all_points.append(xyz)
            all_colors.append(colors)

        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)

        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for (x, y, z), (r, g, b) in zip(all_points, all_colors):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

        print(f"✅ Full 3D point cloud saved to {filepath}")

    def export_bev_pointcloud_to_ply(self, filepath="output/bev_topdown.ply"):
        """
        Export BEV-style oriented 3D point cloud: (x, y_bev, z_height) with RGB
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not self.buffer:
            print("⚠️ No data in buffer.")
            return

        all_points = []
        all_colors = []

        for (xyz, colors) in self.buffer:
            if xyz.shape[1] == 2:
                print("⚠️ Only XZ available. Cannot rotate to BEV without Y.")
                return
            all_points.append(xyz)
            all_colors.append(colors)

        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)

        # Rotate: x → x, y → z_bev, z → y_bev
        x = all_points[:, 0]
        y = all_points[:, 1]
        z = all_points[:, 2]
        x_bev = x
        y_bev = z  # forward becomes vertical
        z_bev = -y  # camera-down becomes BEV up

        points_bev = np.column_stack((x_bev, y_bev, z_bev))

        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_bev)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(points_bev, all_colors):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

        print(f"✅ Saved BEV-oriented point cloud to: {filepath}")

    def export_pointcloud_with_3d_boxes(
        self, points_xyz, colors_rgb,
        box_list,
        filepath="output/pointcloud_with_boxes.ply"
    ):
        """
        points_xyz: N x 3 array of point cloud
        colors_rgb: N x 3 array of RGB values
        box_list: list of (center, size, yaw, color)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        verts = list(points_xyz)
        colors = list(colors_rgb)
        lines = []

        for box_id, (center, size, yaw, color) in enumerate(box_list):
            box_pts, box_lines, box_color = get_bbox_lines(center, size, yaw, color)
            idx_offset = len(verts)
            verts.extend(box_pts)
            colors.extend([box_color] * len(box_pts))
            lines.extend([(i0 + idx_offset, i1 + idx_offset) for i0, i1 in box_lines])

        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element edge {len(lines)}\n")
            f.write("property int vertex1\nproperty int vertex2\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(verts, colors):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")
            for v1, v2 in lines:
                f.write(f"{v1} {v2}\n")

        print(f"✅ Exported 3D point cloud + boxes to {filepath}")

    '''
    def _add_bev_legend(self, canvas):
        legend_x = 10
        legend_y = 20
        cv2.rectangle(canvas, (legend_x - 5, legend_y - 15),
                      (legend_x + 120, legend_y + 60), (0, 0, 0), -1)
        cv2.rectangle(canvas, (legend_x - 5, legend_y - 15),
                      (legend_x + 120, legend_y + 60), (255, 255, 255), 1)

        cv2.putText(canvas, "BEV Legend", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        legend_items = [
            ("Road Surface", (0, 255, 255)),
            ("Vehicles", (0, 0, 255)),
            ("Obstacles", (255, 0, 0)),
            ("Ego Vehicle", (0, 255, 0))
        ]
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + i * 15
            cv2.circle(canvas, (legend_x + 10, y_pos), 5, color, -1)
            cv2.putText(canvas, text, (legend_x + 25, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        ego_size = 10
        cv2.rectangle(canvas,
                      (self.center_x - ego_size, self.center_y - ego_size // 2),
                      (self.center_x + ego_size, self.center_y + ego_size // 2),
                      (0, 255, 0), 2)
                      '''

'''
class EnhancedPseudoLidarBEV:
    def __init__(self, canvas_size=400, scale=5.0, buffer_size=15, max_range=80.0):
        """
        Enhanced pseudo-LiDAR visualization with better point cloud processing and visualization
        
        Args:
            canvas_size: Size of the BEV canvas (square)
            scale: Meters to pixels scaling factor
            buffer_size: Number of frames to accumulate for point cloud
            max_range: Maximum detection range in meters
        """
        self.canvas_size = canvas_size
        self.scale = scale
        self.buffer = deque(maxlen=buffer_size)
        self.max_range = max_range
        self.center_x = canvas_size // 2
        self.center_y = canvas_size - 20  # Leave space at bottom for legend
        
        # Color mapping based on height
        self.height_colormap = cv2.COLORMAP_JET
        
        # Grid parameters
        self.grid_spacing = 10  # meters
        self.grid_color = (50, 50, 50)
        
        # Vehicle history for trails
        self.vehicle_history = defaultdict(lambda: deque(maxlen=20))

    def depth_to_bev_points(self, depth_map, K, frame=None):
        """
        Convert depth map to BEV point cloud with enhanced features
        
        Args:
            depth_map: Depth map from estimator
            K: Camera intrinsics matrix
            frame: Optional RGB frame for color mapping
            
        Returns:
            points_2d: BEV points (x,z) coordinates
            colors: Corresponding colors for each point
        """
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create grid of pixel coordinates
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Only process lower portion of image (road area)
        mask = (ys > h * 0.4) & (depth_map > 0.1) & (depth_map < self.max_range)
        
        # Convert to 3D camera coordinates
        z = depth_map[mask]
        x = (xs[mask] - cx) * z / fx
        y = (ys[mask] - cy) * z / fy
        
        # Color mapping options
        if frame is not None:
            # Use original image colors
            colors = frame[ys[mask], xs[mask]]
        else:
            # Use height-based coloring
            normalized_height = np.clip((y + 2.0) / 4.0, 0, 1)  # Map -2m to +2m to 0-1
            colors = (normalized_height * 255).astype(np.uint8)
            colors = cv2.applyColorMap(colors, self.height_colormap)
            colors = colors.squeeze()
        
        return np.column_stack((x, z)), colors

    def render_bev(self, points, colors=None):
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        # Draw grid
        self._draw_grid(canvas)
        
        if len(points) == 0:
            return canvas
            
        # Convert points to canvas coordinates and ensure they're integers
        canvas_x = (self.center_x + points[:, 0] * self.scale).astype(int)
        canvas_y = (self.center_y - points[:, 1] * self.scale).astype(int)
        
        # Filter points that are within canvas bounds
        valid = (canvas_x >= 0) & (canvas_x < self.canvas_size) & \
                (canvas_y >= 0) & (canvas_y < self.canvas_size)
        canvas_x = canvas_x[valid]
        canvas_y = canvas_y[valid]
        
        # Draw points
        if colors is not None:
            colors = colors[valid]
            for x, y, color in zip(canvas_x, canvas_y, colors):
                cv2.circle(canvas, (x, y), 1, color.tolist(), -1)
        else:
            for x, y in zip(canvas_x, canvas_y):
                cv2.circle(canvas, (x, y), 1, (0, 255, 255), -1)
                
        return canvas

    def _draw_grid(self, canvas):
        """Draw distance grid on BEV canvas"""
        # Vertical lines (left/right)
        for x in range(-50, 51, self.grid_spacing):
            canvas_x = int(self.center_x + x * self.scale)  # Convert to int
            if 0 <= canvas_x < self.canvas_size:
                cv2.line(canvas, (canvas_x, 0), (canvas_x, self.canvas_size), 
                        self.grid_color, 1)
                # Distance label
                cv2.putText(canvas, f"{abs(x)}m", (canvas_x - 15, self.canvas_size - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Horizontal lines (distance)
        for z in range(0, int(self.max_range) + 1, self.grid_spacing):
            canvas_y = int(self.center_y - z * self.scale)  # Convert to int
            if 0 <= canvas_y < self.canvas_size:
                cv2.line(canvas, (0, canvas_y), (self.canvas_size, canvas_y), 
                        self.grid_color, 1)
                # Distance label
                cv2.putText(canvas, f"{z}m", (10, canvas_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def update_vehicle_positions(self, tracked_boxes, categories, depth_map, K):
        """Update vehicle positions for trails"""
        for i, box in enumerate(tracked_boxes):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = y2  # bottom center
            
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                z = depth_map[cy, cx]
                if 0.1 < z < self.max_range:
                    fx = K[0, 0]
                    cx0 = K[0, 2]
                    x = (cx - cx0) * z / fx
                    
                    # Get class-specific color
                    class_id = categories[i]
                    color = colorLabels(class_id) if 'colorLabels' in globals() else (0, 0, 255)
                    
                    # Store position
                    track_id = i  # Should use actual track ID if available
                    self.vehicle_history[track_id].append((x, z, color))

    def draw_vehicle_trails(self, canvas):
        """Draw vehicle movement trails"""
        for track_id, history in self.vehicle_history.items():
            if len(history) < 2:
                continue
                
            points = []
            for x, z, color in history:
                canvas_x = int(self.center_x + x * self.scale)
                canvas_y = int(self.center_y - z * self.scale)
                points.append((canvas_x, canvas_y))
                
                # Draw current position
                cv2.circle(canvas, (canvas_x, canvas_y), 3, color, -1)
            
            # Draw trail
            for i in range(1, len(points)):
                alpha = i / len(points)  # Fade older points
                faded_color = tuple(int(c * alpha) for c in history[i][2])
                cv2.line(canvas, points[i-1], points[i], faded_color, 1)

    def update_and_overlay(self, frame, depth_map, K, tracked_boxes=None, 
                     categories=None, overlay_pos=(10, 10)):
        """
        Enhanced version with better visualization and error handling
        
        Args:
            frame: Input RGB frame
            depth_map: Depth map from estimator
            K: Camera intrinsics
            tracked_boxes: List of bounding boxes
            categories: List of class IDs
            overlay_pos: (x,y) position for overlay
            
        Returns:
            frame: Frame with BEV overlay
        """
        # Generate point cloud
        points, colors = self.depth_to_bev_points(depth_map, K, frame)
        self.buffer.append((points, colors))
        
        # Combine points from buffer
        all_points = np.vstack([p for p, _ in self.buffer]) if self.buffer else np.array([])
        all_colors = np.vstack([c for _, c in self.buffer]) if colors is not None and self.buffer else None
        
        # Render BEV
        bev_map = self.render_bev(all_points, all_colors)
        
        # Update and draw vehicle trails if boxes are provided
        if tracked_boxes is not None and categories is not None:
            self.update_vehicle_positions(tracked_boxes, categories, depth_map, K)
            self.draw_vehicle_trails(bev_map)
        
        # Add legend and info
        self._add_bev_legend(bev_map)
        
        # Calculate overlay position (bottom-left corner)
        h_bev, w_bev = bev_map.shape[:2]
        h_frame, w_frame = frame.shape[:2]
        
        # Position in bottom-left with 10px margin
        x_offset = 10
        y_offset = h_frame - h_bev - 10
        
        # Ensure overlay fits in frame
        if y_offset < 0:
            y_offset = 0
            # Resize if too tall
            scale_factor = h_frame / h_bev
            bev_map = cv2.resize(bev_map, (int(w_bev * scale_factor), h_frame))
            h_bev, w_bev = bev_map.shape[:2]
        
        # Paste the BEV map onto the frame
        try:
            frame[y_offset:y_offset+h_bev, x_offset:x_offset+w_bev] = bev_map
            
            # Add label
            cv2.putText(frame, "BEV Point Cloud", (x_offset, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Warning: BEV overlay failed - {str(e)}")
        
        return frame

    def _add_bev_legend(self, canvas):
        """Add legend to BEV view"""
        legend_x = 10
        legend_y = 20
        
        # Legend background
        cv2.rectangle(canvas, (legend_x-5, legend_y-15), 
                      (legend_x+120, legend_y+60), (0, 0, 0), -1)
        cv2.rectangle(canvas, (legend_x-5, legend_y-15), 
                      (legend_x+120, legend_y+60), (255, 255, 255), 1)
        
        # Title
        cv2.putText(canvas, "BEV Legend", (legend_x, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Items
        legend_items = [
            ("Road Surface", (0, 255, 255)),  # Yellow
            ("Vehicles", (0, 0, 255)),       # Red
            ("Obstacles", (255, 0, 0)),      # Blue
            ("Ego Vehicle", (0, 255, 0))     # Green
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + i * 15
            cv2.circle(canvas, (legend_x + 10, y_pos), 5, color, -1)
            cv2.putText(canvas, text, (legend_x + 25, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw ego vehicle at center bottom
        ego_size = 10
        cv2.rectangle(canvas, 
                     (self.center_x - ego_size, self.center_y - ego_size//2),
                     (self.center_x + ego_size, self.center_y + ego_size//2),
                     (0, 255, 0), 2)
'''

class RollingPseudoLidarBEV:
    def __init__(self, canvas_size=200, scale=5.0, buffer_size=10):
        self.canvas_size = canvas_size
        self.scale = scale
        self.buffer = deque(maxlen=buffer_size)

    def depth_to_bev_points(self, depth_map, K, valid_pixel_mask=None, max_range=50.0):
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # shape: (H, W)

        z = depth_map
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy

        x = x.flatten()
        z = z.flatten()
        ys_flat = ys.flatten()

        # Only keep lower image (e.g., lower 60%)
        mask = (z > 0.1) & (z < max_range) & (ys_flat > h * 0.4)

        return np.vstack((x[mask], z[mask])).T  # shape: [N, 2]

    def render_bev(self, points, canvas_size=200, scale=5.0):
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        center = canvas_size // 2

        for px, pz in points:
            u = int(center + px * scale)
            v = int(canvas_size - pz * scale)
            if 0 <= u < canvas_size and 0 <= v < canvas_size:
                cv2.circle(canvas, (u, v), 1, (0, 255, 255), -1)  # yellow dot

        return canvas

    def update_and_overlay(self, frame, depth_map, K, tracked_boxes=None, categories=None, overlay_pos=(10, -10)):
        bev_points = self.depth_to_bev_points(depth_map, K)
        self.buffer.append(bev_points)

        rolling_points = np.vstack(self.buffer)
        bev_map = self.render_bev(rolling_points)

        # Compute position
        x_offset = overlay_pos[0]
        y_offset = frame.shape[0] + overlay_pos[1] - self.canvas_size
        frame[y_offset:y_offset + self.canvas_size, x_offset:x_offset + self.canvas_size] = bev_map

        # Label
        cv2.putText(frame, "Pseudo-LiDAR Top-Down", (x_offset, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if tracked_boxes is not None:
            for i, box in enumerate(tracked_boxes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = y2  # bottom center of the box

                # Project to camera-space 3D using depth
                if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                    z = depth_map[cy, cx]
                    if 0.1 < z < 80:
                        fx, fy = K[0, 0], K[1, 1]
                        cx0, cy0 = K[0, 2], K[1, 2]
                        x = (cx - cx0) * z / fx
                        # z is depth, x is left-right
                        u = int(self.canvas_size // 2 + x * self.scale)
                        v = int(self.canvas_size - z * self.scale)
                        if 0 <= u < self.canvas_size and 0 <= v < self.canvas_size:
                            color = (0, 0, 255) if categories is None else colorLabels(categories[i])
                            cv2.rectangle(bev_map, (u-3, v-3), (u+3, v+3), color, 1)

        return frame

import open3d as o3d

class TexturedPointCloudExporter:
    def __init__(self, output_dir='output', max_range=80.0):
        self.output_dir = output_dir
        self.max_range = max_range
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, depth_map, rgb_image, K, filename='textured_pointcloud.ply'):
        """
        Generate and save a textured point cloud using depth and RGB image.

        Args:
            depth_map (np.ndarray): Depth map (float32, meters)
            rgb_image (np.ndarray): RGB image (uint8, HxWx3)
            K (np.ndarray): 3x3 camera intrinsics matrix
            filename (str): Name of output .ply file
        """
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create pixel coordinate grid
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        z = depth_map.astype(np.float32)

        # Mask invalid depths
        valid = (z > 0.1) & (z < self.max_range)
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy

        # Form 3D coordinates
        xyz = np.stack((x[valid], -y[valid], z[valid]), axis=-1)  # Flip y for Open3D convention
        colors = rgb_image[valid].astype(np.float32) / 255.0

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Optional: Downsample and remove noise
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Save
        ply_path = os.path.join(self.output_dir, filename)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"✅ Saved textured point cloud: {ply_path}")

        return ply_path

import numpy as np
import cv2
import math
from collections import deque, defaultdict

class Stabilized3DBoundingBox:
    def __init__(self):
        # Intrinsics for 704×576 frame, 70° FOV
        fx, fy = 503, 503
        cx, cy = 352, 288
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])

        # More accurate vehicle dimensions based on actual measurements
        self.vehicle_dimensions = {
            0: {"width": 0.6, "height": 1.7, "length": 0.6},  # person
            2: {"width": 1.8, "height": 1.5, "length": 4.5},  # car
            3: {"width": 0.8, "height": 1.3, "length": 2.2},  # motorbike
            5: {"width": 2.5, "height": 3.2, "length": 12.0}, # bus
            7: {"width": 2.4, "height": 3.0, "length": 8.0},  # truck
        }
        
        # Tracking for stability
        self.object_states = defaultdict(lambda: {
            'depth_history': deque(maxlen=5),
            'yaw_history': deque(maxlen=8),
            #'yaw_kalman': YawKalman(),
            #'pos_kalman': PositionKalman(),
            'position_history': deque(maxlen=5),
            'last_valid_yaw': 0.0,
            'confidence': 0.0
        })

        self.ema_alpha = 0.5  # adjust to control smoothing (0=slow, 1=no smoothing)

        self.object_states = defaultdict(lambda: {
        'depth_ema': None,
        'yaw_ema': None,
        'position_ema': None,
        'dim_ema': None,
        'yaw_history': deque(maxlen=8),
        'position_history': deque(maxlen=5),
        'last_valid_yaw': 0.0,
        'confidence': 0.0,
        'yaw_kalman': YawKalman(),       # ✅ Add this
        'pos_kalman': PositionKalman()   # ✅ Add this too
    })

    def apply_ema_smoothing(self, key, value, track_id):
        """Applies exponential moving average to a property."""
        state = self.object_states[track_id]
        ema_key = f"{key}_ema"

        previous = state.get(ema_key)
        alpha = self.ema_alpha

        if previous is None:
            smoothed = value
        else:
            if isinstance(value, np.ndarray):
                smoothed = alpha * value + (1 - alpha) * previous
            elif isinstance(value, (float, int)):
                smoothed = alpha * value + (1 - alpha) * previous
            elif isinstance(value, tuple):
                smoothed = tuple(alpha * v + (1 - alpha) * p for v, p in zip(value, previous))
            else:
                smoothed = value  # fallback

        state[ema_key] = smoothed
        return smoothed

    def pixel_to_camera_coords(self, x, y, z):
        """Convert pixel coordinates to 3D camera coordinates"""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        return np.array([X, Y, z])

    def project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D image plane"""
        # Add small epsilon to avoid division by zero
        points_3d_safe = points_3d.copy()
        points_3d_safe[:, 2] = np.maximum(points_3d_safe[:, 2], 0.1)
        
        projected = self.K @ points_3d_safe.T
        projected[2] = np.maximum(projected[2], 0.001)  # Avoid division by zero
        projected /= projected[2]
        return projected[:2].T.astype(int)

    def estimate_stable_yaw(self, track_id, current_center, bbox, frame_width=None):
        """Estimate yaw angle with motion or fallback heuristic"""
        state = self.object_states[track_id]
        state['position_history'].append(current_center)

        if len(state['position_history']) < 3:
            return self._estimate_fallback_yaw(bbox, frame_width)  # fallback if not enough history

        positions = list(state['position_history'])
        dx = positions[-1][0] - positions[-3][0]
        dy = positions[-1][1] - positions[-3][1]
        movement_magnitude = math.sqrt(dx**2 + dy**2)

        if movement_magnitude > 3.0:
            # Estimate from motion
            yaw = math.atan2(dy, dx)
            kalman = self.object_states[track_id]['yaw_kalman']
            predicted_yaw = kalman.predict()
            filtered_yaw = kalman.correct(yaw)

            state['yaw_ema'] = filtered_yaw  # for optional smoothing later
            return filtered_yaw
        else:
            # No significant movement — use predicted yaw
            return self.object_states[track_id]['yaw_kalman'].predict()

    def _estimate_fallback_yaw(self, bbox, frame_width=None):
        """
        Estimate yaw based on object's horizontal position in the frame.
        Assumes vehicles face camera direction (depth axis) with slight offset.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2

        if frame_width is None:
            frame_width = self.K[0, 2] * 2  # fallback if not passed

        offset = (cx - frame_width / 2) / frame_width  # -0.5 (left) → +0.5 (right)

        # Map offset to yaw angle range: left face right, right face left
        # -π/8 to π/8 → slight yaw
        yaw_angle = -offset * (math.pi / 4)
        return yaw_angle

    def stabilize_depth(self, track_id, raw_depth):
        """Apply temporal filtering to depth measurements"""
        state = self.object_states[track_id]
        
        # Outlier rejection
        if state['depth_history']:
            recent_depths = list(state['depth_history'])
            median_depth = np.median(recent_depths)
            
            # Reject measurements that are too far from recent median
            if abs(raw_depth - median_depth) > 5.0:
                return median_depth
        
        state['depth_history'].append(raw_depth)
        
        # Use weighted average with more weight on recent measurements
        depths = list(state['depth_history'])
        weights = np.linspace(0.5, 1.0, len(depths))  # More weight on recent
        weighted_depth = np.average(depths, weights=weights)
        
        return weighted_depth

    def get_adaptive_dimensions(self, bbox, class_id, depth):
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        base_dims = self.vehicle_dimensions.get(class_id, {
            "width": 1.8, "height": 1.5, "length": 4.0
        })

        base_w, base_h, base_l = base_dims["width"], base_dims["height"], base_dims["length"]

        depth = max(depth, 0.1)

        expected_w_px = (base_w * self.K[0, 0]) / depth
        expected_h_px = (base_h * self.K[1, 1]) / depth

        scale_w = bbox_width / max(expected_w_px, 1.0)
        scale_h = bbox_height / max(expected_h_px, 1.0)

        # 👉 Boost the scale factors slightly
        scale_w = min(max(scale_w * 1.15, 0.7), 1.8)
        scale_h = min(max(scale_h * 1.2, 0.75), 2.0)

        # 👉 Reduce blend_weight to give more power to adaptive
        blend_weight = 0.2

        width_final = blend_weight * base_w + (1 - blend_weight) * base_w * scale_w
        height_final = blend_weight * base_h + (1 - blend_weight) * base_h * scale_h
        length_final = blend_weight * base_l + (1 - blend_weight) * base_l * scale_w

        return {
            "width": width_final,
            "height": height_final,
            "length": length_final
        }

    def create_3d_corners(self, dimensions, yaw_angle=0):
        """Create 3D corner points for a bounding box"""
        w, h, l = dimensions["width"], dimensions["height"], dimensions["length"]
        
        # Define corners relative to object center (bottom-center)
        corners = np.array([
            # Bottom face (y=0)
            [ w/2,  0,  l/2],  # 0: front-right-bottom
            [-w/2,  0,  l/2],  # 1: front-left-bottom  
            [-w/2,  0, -l/2],  # 2: rear-left-bottom
            [ w/2,  0, -l/2],  # 3: rear-right-bottom
            # Top face (y=-h)
            [ w/2, -h,  l/2],  # 4: front-right-top
            [-w/2, -h,  l/2],  # 5: front-left-top
            [-w/2, -h, -l/2],  # 6: rear-left-top
            [ w/2, -h, -l/2],  # 7: rear-right-top
        ])
        
        # Apply yaw rotation if specified
        if abs(yaw_angle) > 0.1:  # Only rotate if angle is significant
            cos_yaw = math.cos(yaw_angle)
            sin_yaw = math.sin(yaw_angle)
            rotation_matrix = np.array([
                [cos_yaw,  0, sin_yaw],
                [0,        1, 0      ],
                [-sin_yaw, 0, cos_yaw]
            ])
            corners = corners @ rotation_matrix.T
        
        return corners

    def draw_3d_box_enhanced(self, frame, bbox, depth, class_id, track_id, color=(0, 255, 0)):
        """Draw stable 3D bounding box using Kalman-filtered yaw and position"""
        x1, y1, x2, y2 = map(int, bbox)

        # === Raw bottom center of the object (2D anchor point)
        measured_cx = (x1 + x2) // 2
        measured_cy = y2

        # === Kalman-filtered position
        pos_kf = self.object_states[track_id]['pos_kalman']
        pos_kf.predict()
        tracked_cx, tracked_cy = pos_kf.correct(measured_cx, measured_cy)

        # === Kalman-filtered yaw from motion or fallback
        frame_width = frame.shape[1]
        raw_yaw = self.estimate_stable_yaw(track_id, (measured_cx, measured_cy), bbox, frame_width=frame_width)
        yaw_kf = self.object_states[track_id]['yaw_kalman']
        yaw_kf.predict()
        filtered_yaw = yaw_kf.correct(raw_yaw)

        # === Smooth depth (optional, or use Kalman if needed)
        depth_smoothed = self.apply_ema_smoothing("depth", depth, track_id)

        # === Get smoothed physical dimensions
        dims = self.get_adaptive_dimensions(bbox, class_id, depth_smoothed)
        dims_smoothed = self.apply_ema_smoothing("dim", dims, track_id)

        # === Create 3D corners and transform into camera space
        corners_3d = self.create_3d_corners(dims_smoothed, filtered_yaw)
        center_3d = self.pixel_to_camera_coords(tracked_cx, tracked_cy, depth_smoothed)
        corners_world = corners_3d + center_3d

        try:
            # === Project 3D points to 2D image
            corners_2d = self.project_3d_to_2d(corners_world)
            self.object_states[track_id]['last_corners_2d'] = corners_2d

            if len(corners_2d) == 8:
                self.draw_wireframe_3d(frame, corners_2d, color, track_id)

                # Optional: Draw dimensions as text below object
                dim_text = f"{dims_smoothed['width']:.1f}×{dims_smoothed['height']:.1f}×{dims_smoothed['length']:.1f}m"
                cv2.putText(frame, dim_text, (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # Fallback to 2D box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        except Exception as e:
            # Fallback to 2D box if projection fails
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # (Optional) Draw center anchor debug
        cv2.circle(frame, (measured_cx, measured_cy), 3, (0, 0, 255), -1)  # red = raw
        cv2.circle(frame, (tracked_cx, tracked_cy), 3, (0, 255, 0), -1)    # green = filtered

    def draw_3d_box_from_2d_front(self, frame, bbox, depth, yaw, color=(0, 255, 0)):
        """
        Draw a 3D box using the 2D box as the front face and projecting the back face using depth and yaw.
        """
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1

        # Calculate perspective offset based on depth (closer → larger offset)
        offset_factor = max(0.1, 1.0 - depth / 60.0)
        offset_x = int(width * 0.3 * offset_factor)
        offset_y = int(height * 0.3 * offset_factor)

        offset_x = max(10, min(offset_x, 60))
        offset_y = max(10, min(offset_y, 60))

        # Optional: apply yaw direction to skew the offset
        skew_x = int(offset_x * math.cos(yaw))
        skew_y = int(offset_y * math.sin(yaw))

        # Define front face (2D box)
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)

        # Define back face (offset by depth + yaw skew)
        back_tl = (x1 + skew_x, y1 - offset_y)
        back_tr = (x2 + skew_x, y1 - offset_y)
        back_br = (x2 + skew_x, y2 - offset_y)
        back_bl = (x1 + skew_x, y2 - offset_y)

        # Draw faces
        overlay = frame.copy()

        # Front face
        cv2.rectangle(frame, front_tl, front_br, color, 2)

        # Connect faces
        cv2.line(frame, front_tl, back_tl, color, 1)
        cv2.line(frame, front_tr, back_tr, color, 1)
        cv2.line(frame, front_br, back_br, color, 1)
        cv2.line(frame, front_bl, back_bl, color, 1)

        # Back face
        cv2.line(frame, back_tl, back_tr, color, 1)
        cv2.line(frame, back_tr, back_br, color, 1)
        cv2.line(frame, back_br, back_bl, color, 1)
        cv2.line(frame, back_bl, back_tl, color, 1)

        # Shaded top face
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)

        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def get_last_projected_corners(self, track_id):
        """Retrieve the last projected 2D corners of the 3D box for this object"""
        return self.object_states[track_id].get('last_corners_2d', None)

    def draw_wireframe_3d(self, frame, corners_2d, color, track_id):
        """Draw 3D wireframe with confidence-based alpha"""
        if len(corners_2d) < 8:
            return
            
        state = self.object_states[track_id]
        confidence = state['confidence']
        
        # Adjust line thickness and alpha based on confidence
        thickness = max(1, int(2 * confidence))
        alpha = 0.3 + 0.7 * confidence
        
        # Create overlay for alpha blending
        overlay = frame.copy()
        
        # Define edges for a 3D box
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face  
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw edges
        for start_idx, end_idx in edges:
            if start_idx < len(corners_2d) and end_idx < len(corners_2d):
                start_pt = tuple(corners_2d[start_idx].astype(int))
                end_pt = tuple(corners_2d[end_idx].astype(int))
                cv2.line(overlay, start_pt, end_pt, color, thickness)
        
        # Blend with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw corner points for better visibility
        for corner in corners_2d:
            corner_pt = tuple(corner.astype(int))
            cv2.circle(frame, corner_pt, 2, color, -1)

    def cleanup_old_tracks(self, active_track_ids):
        """Remove tracking data for inactive tracks"""
        inactive_ids = set(self.object_states.keys()) - set(active_track_ids)
        for track_id in inactive_ids:
            del self.object_states[track_id]

def initialize_deepsort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # Initialize the DeepSort tracker
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
        )

    return deepsort

deepsort = initialize_deepsort()
data_deque = {}
speed_data = {}
distance_data = {}
frame_times = {}

def classNames():
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    return cocoClassNames

className = classNames()

def colorLabels(classid):
    if classid == 0: #person
        color = (85, 45, 255)
    elif classid == 2: #car
        color = (222, 82, 175)
    elif classid == 3: #Motorbike
        color = (0, 204, 255)
    elif classid == 5: #Bus
        color = (0,149,255)
    else:
        color = (200, 100,0)
    return tuple(color)

previous_depths = {}

def calculate_speed_from_depth(track_id, current_depth, fps):
    """
    Estimate vehicle speed based on depth change over time.
    """
    speed_kmh = 0
    if track_id in previous_depths:
        prev_depth = previous_depths[track_id]
        delta_depth = abs(current_depth - prev_depth)
        time_interval = 1.0 / fps
        speed_m_s = delta_depth / time_interval
        speed_kmh = speed_m_s * 3.6  # convert to km/h
    previous_depths[track_id] = current_depth
    return speed_kmh

def calculate_safe_distance(speed_kmh):
    """
    Calculate safe following distance based on speed
    Formula: Safe distance = (speed in km/h / 10) * 3 + 3 meters
    """
    if speed_kmh < 10:
        return 10  # Minimum safe distance
    return (speed_kmh / 10) * 3 + 3

def get_distance_color(actual_distance, safe_distance):
    """
    Get color based on safe distance comparison
    Green: Safe, Yellow: Caution, Red: Dangerous
    """
    ratio = actual_distance / safe_distance if safe_distance > 0 else 1
    
    if ratio >= 1.2:  # 20% more than safe distance
        return (0, 255, 0)  # Green - Safe
    elif ratio >= 0.8:  # 80% of safe distance
        return (0, 255, 255)  # Yellow - Caution
    else:
        return (0, 0, 255)  # Red - Dangerous

def calculate_distance_between_vehicles(center1, center2, pixels_per_meter=20):
    """Calculate distance between two vehicles in meters"""
    distance_pixels = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance_pixels / pixels_per_meter

def is_vehicle(class_id):
    """Check if the detected object is a vehicle"""
    vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck
    return class_id in vehicle_classes

def dynamic_calibrate_pixels_per_meter(detections, class_ids):
    """
    Calibrate pixels per meter using a car's bounding box width.
    """
    for det, cls in zip(detections, class_ids):
        if int(cls) == 2:  # class 2 = car
            x1, y1, x2, y2 = map(int, det)
            pixel_width = abs(x2 - x1)
            if pixel_width > 0:
                return pixel_width / 1.8  # assuming average car width is 1.8 meters
    return None  # fallback if no car is detected

def draw_statistics_panel(frame, stats):
    x, y, w, h = 10, 10, 220, 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    lines = [
        f"Cars: {stats['car']}",
        f"Motorbikes: {stats['motorbike']}",
        f"Trucks: {stats['truck']}",
        f"Persons: {stats['person']}",
        f"Alerts: {stats['alerts']}"
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 10, y + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

import math

def draw_fake_3d_box_with_orientation(frame, x1, y1, x2, y2, direction=None, color=(255, 0, 0), scale=0.25):
    """
    Draws a fake 3D cuboid with tilt based on estimated vehicle direction.
    - direction: angle in radians (from atan2)
    - scale: depth shift scaling
    """
    box_width = x2 - x1
    box_height = y2 - y1

    # FRONT face (camera-facing)
    front_tl = (x1, y1)
    front_tr = (x2, y1)
    front_bl = (x1, y2)
    front_br = (x2, y2)

    # If direction is None, draw regular 3D box
    if direction is None:
        dx_left = int(box_width * scale)
        dx_right = int(box_width * scale * 0.6)
        dy = int(box_height * scale)

        direction = math.atan2(-dy, -(dx_left + dx_right) / 2)  # simulate back tilt

    # Project rear face based on heading direction
    depth = int(box_width * scale)
    dx = int(math.cos(direction) * depth)
    dy = int(math.sin(direction) * depth)

    # BACK face (shifted from front face)
    back_tl = (front_tl[0] - dx, front_tl[1] - dy)
    back_tr = (front_tr[0] - dx, front_tr[1] - dy)
    back_bl = (front_bl[0] - dx, front_bl[1] - dy)
    back_br = (front_br[0] - dx, front_br[1] - dy)

    # Draw edges
    cv2.line(frame, front_tl, front_tr, color, 1)
    cv2.line(frame, front_tr, front_br, color, 1)
    cv2.line(frame, front_br, front_bl, color, 1)
    cv2.line(frame, front_bl, front_tl, color, 1)

    cv2.line(frame, back_tl, back_tr, color, 1)
    cv2.line(frame, back_tr, back_br, color, 1)
    cv2.line(frame, back_br, back_bl, color, 1)
    cv2.line(frame, back_bl, back_tl, color, 1)

    cv2.line(frame, front_tl, back_tl, color, 1)
    cv2.line(frame, front_tr, back_tr, color, 1)
    cv2.line(frame, front_bl, back_bl, color, 1)
    cv2.line(frame, front_br, back_br, color, 1)
'''
def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0,0), fps=30, pixels_per_meter=20):
    global calibrated_pixels_per_meter
    height, width, _ = frame.shape
    camera_position = (width // 2, height)
    too_close_warning = False
    stats = {'car': 0, 'motorbike': 0, 'truck': 0, 'person': 0, 'alerts': 0}

    # Dynamic calibration
    if calibrated_pixels_per_meter is None and len(bbox_xyxy) > 0:
        calibrated = dynamic_calibrate_pixels_per_meter(bbox_xyxy, categories)
        calibrated_pixels_per_meter = calibrated if calibrated else pixels_per_meter

    ppm = calibrated_pixels_per_meter or pixels_per_meter
    vehicle_positions = {}

    # Cleanup old IDs
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)
            speed_data.pop(key, None)
            frame_times.pop(key, None)

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        box_color = colorLabels(cat)

        # Update stats
        if cat == 2:
            stats['car'] += 1
        elif cat == 3:
            stats['motorbike'] += 1
        elif cat in [5, 7]:
            stats['truck'] += 1
        elif cat == 0:
            stats['person'] += 1

        # Track center trail
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        data_deque[id].appendleft(center)

        # Estimate speed & direction
        speed = 0
        direction_angle = None
        if is_vehicle(cat):
            speed = calculate_speed(id, center, fps, ppm)
            vehicle_positions[id] = {'center': center, 'speed': speed, 'class': cat}

            if len(data_deque[id]) >= 2:
                (x_prev, y_prev) = data_deque[id][1]
                (x_now, y_now) = data_deque[id][0]
                dx = x_now - x_prev
                dy = y_now - y_prev
                if dx**2 + dy**2 > 25:  # only apply if movement > 5 pixels
                    yaw_angle = math.atan2(dy, dx)

        # Draw bounding box or 3D box
        if is_vehicle(cat):
            draw_fake_3d_box_with_orientation(frame, x1, y1, x2, y2, direction=direction_angle, color=box_color)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

        # Label with speed
        label = f"{id}:{className[cat]}"
        if is_vehicle(cat) and speed > 0:
            label += f" {speed:.1f}km/h"

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, box_color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

        # Center dot
        cv2.circle(frame, center, 2, (0, 255, 0), cv2.FILLED)

        # Trails
        if draw_trails:
            for j in range(1, len(data_deque[id])):
                if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
                cv2.line(frame, data_deque[id][j - 1], data_deque[id][j], box_color, thickness)

        # Distance from camera
        if is_vehicle(cat):
            actual_distance = calculate_distance_between_vehicles(camera_position, center, ppm)
            if actual_distance <= 50:
                safe_distance = calculate_safe_distance(speed)
                line_color = get_distance_color(actual_distance, safe_distance)
                if actual_distance < safe_distance:
                    stats['alerts'] += 1
                    too_close_warning = True
                # Line from camera to vehicle
                cv2.line(frame, camera_position, center, line_color, 1)
                # Distance label
                mid_x = (camera_position[0] + center[0]) // 2
                mid_y = (camera_position[1] + center[1]) // 2
                cv2.putText(frame, f"{actual_distance:.1f}m", (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

    # Draw overlay
    #draw_statistics_panel(frame, stats)

    if too_close_warning:
        cv2.putText(frame, "WARNING: Vehicle too close!", (30, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    return frame
    '''

def draw_legend(frame):
    """Draw legend for distance color coding"""
    height, width = frame.shape[:2]
    
    # Legend position (top-right corner)
    legend_x = width - 200
    legend_y = 30
    
    # Draw legend background
    cv2.rectangle(frame, (legend_x - 10, legend_y - 25), (legend_x + 190, legend_y + 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (legend_x - 10, legend_y - 25), (legend_x + 190, legend_y + 70), (255, 255, 255), 2)
    
    # Legend title
    cv2.putText(frame, "Distance Safety", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Legend items
    legend_items = [
        ("Safe", (0, 255, 0)),
        ("Caution", (0, 255, 255)),
        ("Dangerous", (0, 0, 255))
    ]
    
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + 20 + i * 15
        cv2.circle(frame, (legend_x + 10, y_pos), 5, color, -1)
        cv2.putText(frame, text, (legend_x + 25, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def depth_correction(bbox, class_id, raw_depth, intrinsics, label=None, fx_override=True):
    """
    Stabilizes and bounds raw depth for 3D box projection.

    Args:
        bbox: (x1, y1, x2, y2)
        class_id: int
        raw_depth: float (meters)
        intrinsics: numpy array (3x3)
        label: optional class name (e.g., "car")
        fx_override: bool, whether to override unrealistic fx-derived depth

    Returns:
        corrected_depth: float
    """
    x1, _, x2, _ = bbox
    bbox_px_width = x2 - x1
    fx = intrinsics[0, 0]

    # Real object widths
    class_widths = {
        2: 1.8,  # car
        3: 0.8,  # motorbike
        5: 2.5,  # bus
        7: 2.4,  # truck
    }
    real_width = class_widths.get(class_id, 1.8)

    # Estimate fx from observed size and depth
    if raw_depth > 0.1:
        estimated_fx = (bbox_px_width * raw_depth) / real_width
    else:
        estimated_fx = fx  # fallback

    # Clamp raw depth by class
    class_bounds = {
        2: (3.0, 55.0),  # car
        3: (2.0, 40.0),  # motorbike
        5: (5.0, 70.0),  # bus
        7: (4.0, 65.0),  # truck
    }
    min_d, max_d = class_bounds.get(class_id, (2.0, 60.0))
    corrected_depth = np.clip(raw_depth, min_d, max_d)

    # Check for inconsistent fx
    if fx_override and estimated_fx > fx * 2:
        #print(f"⚠️ Depth too large for {label or class_id}: fx est = {estimated_fx:.1f}, shrinking depth {raw_depth:.1f}m → ", end="")
        corrected_depth *= 0.6  # Reduce depth to re-expand projected box
        #print(f"{corrected_depth:.1f}m")

    return corrected_depth

def get_bbox_lines(center, size, yaw, color=(255, 0, 0)):
    """
    Return 3D bounding box vertices and lines as point+edge format for .ply
    center: (x, y, z)
    size: (l, w, h)
    yaw: rotation around Z
    color: (r, g, b)
    """
    l, w, h = size
    x, y, z = center
    corners = np.array([
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])
    # Rotation matrix
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    rotated = (R @ corners.T).T + np.array(center)

    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
        (4, 5), (5, 6), (6, 7), (7, 4),  # top square
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical lines
    ]
    return rotated, lines, color

def export_pointcloud_with_3d_boxes(
        points_xyz, colors_rgb,
        box_list,
        filepath="output/pointcloud_with_boxes.ply"
    ):
        """
        points_xyz: N x 3 array of point cloud
        colors_rgb: N x 3 array of RGB values
        box_list: list of (center, size, yaw, color)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        verts = list(points_xyz)
        colors = list(colors_rgb)
        lines = []

        for box_id, (center, size, yaw, color) in enumerate(box_list):
            box_pts, box_lines, box_color = get_bbox_lines(center, size, yaw, color)
            idx_offset = len(verts)
            verts.extend(box_pts)
            colors.extend([box_color] * len(box_pts))
            lines.extend([(i0 + idx_offset, i1 + idx_offset) for i0, i1 in box_lines])

        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element edge {len(lines)}\n")
            f.write("property int vertex1\nproperty int vertex2\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(verts, colors):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")
            for v1, v2 in lines:
                f.write(f"{v1} {v2}\n")

        print(f"✅ Exported 3D point cloud + boxes to {filepath}")

# Modified draw_boxes_with_depth function to use the improved 3D bounding box
def draw_boxes_with_depth_improved(frame, bbox_xyxy, depth_estimator, bbox_3d, identities=None,
                                 categories=None, draw_trails=True, data_deque=None, fps=30):
    """Improved version with stable 3D bounding boxes"""
    stats = {'car': 0, 'motorbike': 0, 'truck': 0, 'person': 0, 'alerts': 0}
    depth_map, depth_vis = depth_estimator.estimate_depth(frame)
    # Resize depth map to small overlay size
    depth_overlay = cv2.resize(depth_vis, (160, 120))
    if len(depth_overlay.shape) == 2:  # if grayscale
        depth_overlay = cv2.cvtColor(depth_overlay, cv2.COLOR_GRAY2BGR)

    # Overlay position: top-right
    h_overlay, w_overlay = depth_overlay.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    x_offset = w_frame - w_overlay - 10
    y_offset = 10

    # Paste depth map into top-right corner
    frame[y_offset:y_offset + h_overlay, x_offset:x_offset + w_overlay] = depth_overlay
    height, width, _ = frame.shape
    camera_position = (width // 2, height)

    cv2.putText(frame, "Depth Map", (x_offset, y_offset - 5),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    active_track_ids = []

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        bottom_y = y2
        center = (center_x, bottom_y)

        id = int(identities[i]) if identities is not None else 0
        cat = int(categories[i]) if categories is not None else 0
        active_track_ids.append(id)
        
        label = className[cat] if 'className' in globals() else f"class_{cat}"
        color = colorLabels(cat) if 'colorLabels' in globals() else (0, 255, 0)

        # ✅ Always track trail position, regardless of depth validity
        if draw_trails and data_deque is not None:
            if id not in data_deque:
                data_deque[id] = deque(maxlen=64)
            data_deque[id].appendleft(center)

        # Clamp coordinates to image boundaries
        bottom_y = max(0, min(depth_map.shape[0] - 1, bottom_y))
        center_x = max(0, min(depth_map.shape[1] - 1, center_x))

        # Extract depth with better sampling
        patch_size = 3
        y_start = max(0, bottom_y - patch_size//2)
        y_end = min(depth_map.shape[0], bottom_y + patch_size//2 + 1)
        x_start = max(0, center_x - patch_size//2)
        x_end = min(depth_map.shape[1], center_x + patch_size//2 + 1)
        
        depth_patch = depth_map[y_start:y_end, x_start:x_end]
        raw_depth = float(np.median(depth_patch))

        # Skip obviously bad depths
        if not (0.1 < raw_depth < 80):
            continue

        # ✅ Apply depth correction (this is the key step!)
        depth_distance = depth_correction(
            (x1, y1, x2, y2),
            cat,
            raw_depth,
            bbox_3d.K,
            label=label,
            fx_override=True
        )

        # ✅ Now safely draw the trail (only if center was added before)
        if draw_trails and data_deque:
            for j in range(1, len(data_deque[id])):
                if data_deque[id][j] and data_deque[id][j - 1]:
                    thickness = max(1, int(np.sqrt(64 / float(j + 1)) * 2))
                    cv2.line(frame, data_deque[id][j - 1], data_deque[id][j], color, thickness)

        # Log to CSV
        fx_nominal = bbox_3d.K[0, 0]
        bbox_px_width = x2 - x1
        if raw_depth > 0.1:
            fx_estimated = (bbox_px_width * raw_depth) / bbox_3d.vehicle_dimensions.get(cat, {"width": 1.8})["width"]
        else:
            fx_estimated = fx_nominal

        with open(depth_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                getattr(draw_boxes_with_depth_improved, 'frame_id', -1),
                id,
                cat,
                label,
                bbox_px_width,
                round(raw_depth, 2),
                round(depth_distance, 2),
                round(fx_estimated, 2),
                fx_nominal
            ])

        # === Debug Focal Length Estimation ===
        if cat in bbox_3d.vehicle_dimensions and depth_distance > 1.0:
            bbox_px_width = x2 - x1
            real_width_m = bbox_3d.vehicle_dimensions[cat]["width"]
            
            estimated_fx = (bbox_px_width * raw_depth) / real_width_m
            #print(f"[fx est] ID {id} | {label} | px: {bbox_px_width}px | depth: {raw_depth:.1f}m | fx ≈ {estimated_fx:.1f}")


        # Update statistics
        if cat == 2: stats['car'] += 1
        elif cat == 3: stats['motorbike'] += 1
        elif cat in [5, 7]: stats['truck'] += 1
        elif cat == 0: stats['person'] += 1

        speed = calculate_speed_from_depth(id, depth_distance, fps)

        # Draw enhanced 3D bounding box
        if cat in [0, 2, 3, 5, 7]:  # Vehicles and pedestrians
            # === Kalman-tracked yaw smoothing
            frame_width = frame.shape[1]
            raw_yaw = bbox_3d.estimate_stable_yaw(id, (center_x, bottom_y), (x1, y1, x2, y2), frame_width=frame_width)
            yaw_kf = bbox_3d.object_states[id]['yaw_kalman']
            yaw_kf.predict()
            yaw_smoothed = yaw_kf.correct(raw_yaw)
            bbox_3d.draw_3d_box_from_2d_front(frame, (x1, y1, x2, y2), depth_distance, yaw_smoothed, color)
            #bbox_3d.draw_3d_box_enhanced(frame, (x1, y1, x2, y2), depth_distance, cat, id, color)
        else:
            # Draw 2D box for non-vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add labels and distance information
        #text = f"ID:{id} {label}"
        # Build main label
        main_text = f"{id}:{label}"
        if label in ['car', 'motorbike', 'bus', 'truck'] and speed > 0:
            main_text += f" {speed:.1f}km/h"

        # Draw background for main label
        (text_width, text_height), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        #cv2.putText(frame, main_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Use projected 3D top face to place label precisely
        corners_2d = bbox_3d.get_last_projected_corners(id)
        if corners_2d is not None and len(corners_2d) >= 8:
            top_face_y = np.min([pt[1] for pt in corners_2d[4:8]])  # top face
            label_y = max(0, int(top_face_y) - 5)
        else:
            label_y = max(0, y1 - 8)  # fallback

        cv2.putText(frame, main_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Safety status below it (if applicable)
        #safety = analyze_safety_with_depth(depth_distance, frame, center)
        #if safety['risk_level'] == 'high':
        #    stats['alerts'] += 1
        #    danger_text = "DANGER"
        #    text_color = (0, 0, 255)
        #elif safety['risk_level'] == 'medium':
        #    danger_text = "CAUTION"
        #    text_color = (0, 255, 255)
        #else:
        #    danger_text = ""
        #    text_color = (255, 255, 255)

        # Draw second line if needed
        #if danger_text:
        #    (tw, th), _ = cv2.getTextSize(danger_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #    cv2.rectangle(frame, (x1, y1 - text_height - th - 8), (x1 + tw, y1 - text_height - 5), color, -1)
        #    cv2.putText(frame, danger_text, (x1, y1 - text_height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Draw distance line and measurement
        #cv2.line(frame, camera_position, center, safety['color'], 1)
        #mid_x = (camera_position[0] + center[0]) // 2
        #mid_y = (camera_position[1] + center[1]) // 2
        #cv2.putText(frame, f"{depth_distance:.1f}m", (mid_x, mid_y),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.4, safety['color'], 1)

    # Cleanup old tracking data
    bbox_3d.cleanup_old_tracks(active_track_ids)
    
    # Draw statistics panel (implement draw_statistics_panel if not available)
    #if 'draw_statistics_panel' in globals():
    #    draw_statistics_panel(frame, stats)

    return frame, stats
'''
def analyze_safety_with_depth(distance, frame, vehicle_center):
    """
    Assess danger based on depth-based distance.
    """
    if distance < 3.5:
        risk_level = 'high'
        color = (0, 0, 255)
    elif distance < 10:
        risk_level = 'medium'
        color = (0, 255, 255)
    else:
        return {'risk_level': 'low', 'distance': distance, 'color': (0, 255, 0)}

    radius = int(20 + 100 / (distance + 0.1))
    cv2.circle(frame, vehicle_center, radius, color, 1)
    return {
        'risk_level': risk_level,
        'distance': distance,
        'color': color
    }
    '''
'''
def process_frame_with_yolop(frame, yolop_model):
    """
    Run YOLOP on a frame and return lane/drivable area masks.
    """
    # Preprocess
    img = transform(frame).unsqueeze(0).to(device)
    if half:  # FP16 support
        img = img.half()

    # Inference
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = yolop_model(img)

    # Post-process segmentation masks
    da_seg_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy()  # Drivable area
    ll_seg_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy()  # Lane lines

    return da_seg_mask, ll_seg_mask
    '''
#pseudo_lidar_bev = RollingPseudoLidarBEV(canvas_size=200, scale=5.0, buffer_size=10)
pseudo_lidar_bev = EnhancedPseudoLidarBEV(canvas_size=200, scale=5.0, buffer_size=15)
textured_exporter = TexturedPointCloudExporter(output_dir='/teamspace/studios/this_studio/yolov9/output/pointclouds')
os.makedirs('/teamspace/studios/this_studio/yolov9/output/pointclouds', exist_ok=True)


#@smart_inference_mode()
@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        half=False,
        dnn=False,
        vid_stride=1,
        save_frames=False,
        pixels_per_meter=20,
        draw_trails=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create frames directory if needed
    frames_dir = None
    if save_frames:
        frames_dir = save_dir / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv9 model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Initialize YOLOP
    yolop_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    yolop_model = get_net(cfg)
    yolop_weights = torch.load("../YOLOP/weights/End-to-end.pth", map_location=device)
    yolop_model.load_state_dict(yolop_weights["state_dict"])
    yolop_model.to(device).eval()

    # Initialize other components
    depth_estimator = DepthEstimator("Intel/dpt-large")
    bbox_3d = Stabilized3DBoundingBox()
    deepsort = initialize_deepsort()

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get FPS for speed calculation
    fps = 30
    if hasattr(dataset, 'cap') and dataset.cap is not None:
        fps = dataset.cap.get(cv2.CAP_PROP_FPS)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    frame_count = 0
    
    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # YOLOv9 Inference
        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
            
            # Extract predictions from complex output structure
            if isinstance(pred, (list, tuple)):
                # Try common output structures
                if len(pred) > 1 and isinstance(pred[1], torch.Tensor):
                    pred = pred[1]  # Common in some YOLO variants
                elif len(pred) > 0:
                    pred = pred[0]  # Fallback to first element
            elif hasattr(pred, 'pred'):  # Handle potential wrapper objects
                pred = pred.pred
            
            # Final attempt to get tensor
            while isinstance(pred, (list, tuple)) and len(pred) > 0:
                pred = pred[0]
            
            if not isinstance(pred, torch.Tensor):
                raise RuntimeError(f"Cannot extract predictions. Final output: {pred}")

        # Now proceed with NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy()
            
            # Create visualization frame
            visualization_frame = im0.copy()

            # Run YOLOP every 3 frames for better performance
            if frame_count % 1 == 0:
                yolop_input = yolop_transform(im0).unsqueeze(0).to(device)
                if half:
                    yolop_input = yolop_input.half()
                
                with torch.no_grad():
                    _, da_seg_out, ll_seg_out = yolop_model(yolop_input)
                
                da_seg_mask = torch.argmax(da_seg_out, 1).squeeze().cpu().numpy().astype(np.uint8)
                ll_seg_mask = torch.argmax(ll_seg_out, 1).squeeze().cpu().numpy().astype(np.uint8)
                
                if da_seg_mask.shape != im0.shape[:2]:
                    da_seg_mask = cv2.resize(da_seg_mask, (im0.shape[1], im0.shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
                    ll_seg_mask = cv2.resize(ll_seg_mask, (im0.shape[1], im0.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                
                # Apply segmentation overlay
                segmentation_overlay = np.zeros_like(visualization_frame)
                segmentation_overlay[da_seg_mask == 1] = [0, 255, 0]  # Drivable area
                segmentation_overlay[ll_seg_mask == 1] = [255, 0, 0]   # Lane lines
                cv2.addWeighted(segmentation_overlay, 0.3, visualization_frame, 0.7, 0, visualization_frame)

            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                
                # DeepSORT tracking
                xywh_bboxs = []
                confs = []
                oids = []
                outputs = []
                
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    bbox_width = abs(x1-x2)
                    bbox_height = abs(y1-y2)
                    xcycwh = [cx, cy, bbox_width, bbox_height]
                    xywh_bboxs.append(xcycwh)
                    conf = math.ceil(conf*100)/100
                    confs.append(conf)
                    classNameInt = int(cls)
                    oids.append(classNameInt)
                    
                xywhs = torch.tensor(xywh_bboxs)
                confss = torch.tensor(confs)
                outputs = deepsort.update(xywhs.cpu(), confss.cpu(), oids, imc)

                # Define stats by default so it exists no matter what
                stats = {'car': 0, 'motorbike': 0, 'truck': 0, 'person': 0, 'alerts': 0}
                
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    
                    # Draw 3D boxes and tracking
                    try:
                        visualization_frame, stats = draw_boxes_with_depth_improved(
                            visualization_frame,
                            bbox_xyxy,
                            depth_estimator,
                            bbox_3d,
                            identities=identities,
                            categories=object_id,
                            draw_trails=False,
                            data_deque=data_deque,
                            fps=fps
                        )
                    except Exception as e:
                        #print(f"⚠️ draw_boxes_with_depth_improved() failed: {e}")
                        stats = {'car': 0, 'motorbike': 0, 'truck': 0, 'person': 0, 'alerts': 0}

            # Add depth map overlay
            # Estimate depth from current frame
            depth_map, depth_vis = depth_estimator.estimate_depth(im0)

            #print(f"Depth min: {depth_map.min():.2f}, max: {depth_map.max():.2f}, mean: {depth_map.mean():.2f}")
            print("Depth map dtype:", depth_map.dtype, "min:", depth_map.min(), "max:", depth_map.max())

            # Overlay small depth image in top-right corner
            depth_overlay = cv2.resize(depth_vis, (160, 120))
            visualization_frame[10:130, -170:-10] = depth_overlay

            pseudo_lidar_bev.clean_frame = im0.copy()

            # Overlay rolling top-down pseudo-lidar map in bottom-left
            visualization_frame = pseudo_lidar_bev.update_and_overlay(
                visualization_frame,
                depth_map,
                bbox_3d.K,
                tracked_boxes=bbox_xyxy if len(outputs) > 0 else None,
                categories=object_id if len(outputs) > 0 else None,
                overlay_pos=(10, visualization_frame.shape[0] - 10)  # bottom-left
            )
            cv2.putText(visualization_frame, "Depth Map", (visualization_frame.shape[1] - 170, 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw statistics panel
            #stats = {'car': 0, 'motorbike': 0, 'truck': 0, 'person': 0, 'alerts': 0}
            draw_statistics_panel(visualization_frame, stats)

            os.makedirs("/teamspace/studios/this_studio/yolov9/output", exist_ok=True)

            pseudo_lidar_bev.export_to_ply("/teamspace/studios/this_studio/yolov9/output/bev_cloud.ply")
            pseudo_lidar_bev.export_full_pointcloud_to_ply("/teamspace/studios/this_studio/yolov9/output/pseudo_lidar_full.ply")
            pseudo_lidar_bev.export_bev_pointcloud_to_ply("/teamspace/studios/this_studio/yolov9/output/pseudo_lidar_bev.ply")
            frame_name = f"frame_{frame:06d}_textured.ply"
            textured_exporter.export(depth_map, im0, bbox_3d.K, filename=frame_name)

            # Stream results
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), visualization_frame.shape[1], visualization_frame.shape[0])
                cv2.imshow(str(p), visualization_frame)
                cv2.waitKey(1)
                
            # Save results
            if save_img:
                if save_frames and frames_dir is not None:
                    frame_filename = f"frame_{frame:06d}.jpg"
                    frame_path = str(frames_dir / frame_filename)
                    cv2.imwrite(frame_path, visualization_frame)
                
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, visualization_frame.shape[1], visualization_frame.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(visualization_frame)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
    if update:
        strip_optimizer(weights[0])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--draw-trails', action='store_true', help='draw tracking trails')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--save-frames', action='store_true', help='save individual frames with detections')
    parser.add_argument('--pixels-per-meter', type=int, default=20, help='calibration factor for speed calculation')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)