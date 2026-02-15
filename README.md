Below is a **comprehensive GitHub repository structure and README** for your Autonomous Perception System based on your uploaded `av_perception_system.py` implementation.

This includes:

* Full project architecture
* Flow diagram
* Mathematical formulation
* Model sources & citations
* Pseudo-LiDAR equations
* 3D box projection math
* Kalman filtering equations
* BEV transformation math
* Installation instructions
* Research-grade documentation

---

# 🚗 AV-Multimodal-Perception-System

> A modular autonomous perception stack combining detection, tracking, segmentation, monocular depth estimation, pseudo-LiDAR reconstruction, stabilized 3D bounding boxes, and BEV mapping.

---

# 📁 Repository Structure

```
AV-Multimodal-Perception-System/
│
├── av_perception_system.py
├── models/
│   ├── yolo/
│   ├── yolop/
│   ├── sam2/
│   ├── depth_anything/
│
├── configs/
│   ├── deepsort.yaml
│   ├── camera_intrinsics.yaml
│
├── output/
│   ├── bev_topdown.ply
│   ├── pointcloud_with_boxes.ply
│   ├── textured_pointcloud.ply
│
├── depth_log.csv
├── requirements.txt
├── README.md
└── docs/
    ├── system_diagram.png
    ├── equations.md
```

---

# 🧠 System Overview

This system integrates:

| Module                            | Model Used                          |
| --------------------------------- | ----------------------------------- |
| Object Detection                  | YOLOv9                              |
| Object Tracking                   | DeepSORT                            |
| Drivable Area + Lane Segmentation | YOLOP                               |
| Instance Segmentation             | SAM 2                               |
| Depth Estimation                  | Depth Anything V3 (DA3Metric-Large) |
| 3D Box Stabilization              | Kalman Filter                       |
| Pseudo-LiDAR BEV                  | Custom EnhancedPseudoLidarBEV       |
| 3D Point Cloud Export             | Open3D                              |

---

# 🔁 System Flow Diagram

```
                RGB Frame
                    │
                    ▼
         ┌──────────────────┐
         │   YOLOv9         │
         │  Object Detect   │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │   DeepSORT       │
         │   Tracking       │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Depth Anything   │
         │ Monocular Depth  │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Pseudo-LiDAR     │
         │ 3D Reconstruction│
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ 3D Box Estimator │
         │ + Kalman Filter  │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ BEV Renderer     │
         │ + PLY Export     │
         └──────────────────┘
```

---

# 📐 Mathematical Formulation

---

## 1️⃣ Camera Projection Model

Camera intrinsic matrix:

[
K =
\begin{bmatrix}
f_x & 0 & c_x \
0 & f_y & c_y \
0 & 0 & 1
\end{bmatrix}
]

Pixel to 3D camera coordinate:

[
X = \frac{(u - c_x) Z}{f_x}
]

[
Y = \frac{(v - c_y) Z}{f_y}
]

[
Z = \text{Depth value}
]

Used in:

* `pixel_to_camera_coords()`
* `depth_to_bev_points()`
* 3D box projection

---

## 2️⃣ Monocular Depth Estimation

Using:

* Depth Anything V3 (metric model)

Depth inference:

[
D = f_\theta(I)
]

Where:

* ( I ) = RGB image
* ( \theta ) = pretrained transformer weights
* ( D ) = dense depth map

---

## 3️⃣ Pseudo-LiDAR Reconstruction

Convert depth map to 3D points:

[
\mathbf{P} = (X, Y, Z)
]

Filter by:

[
0.1 < Z < 80m
]

Points stacked into rolling buffer:

[
P_{global} = \bigcup_{t-k}^{t} P_t
]

---

## 4️⃣ BEV Transformation

Top-down projection:

[
x_{bev} = center_x + X \cdot scale
]

[
y_{bev} = center_y - Z \cdot scale
]

Height-based coloring:

[
color = \text{colormap}(Y)
]

---

## 5️⃣ 3D Bounding Box Estimation

Vehicle dimensions:

[
w, h, l
]

3D corners:

[
C = \begin{bmatrix}
\pm w/2 & 0 & \pm l/2 \
\pm w/2 & -h & \pm l/2
\end{bmatrix}
]

Yaw rotation:

[
R_y =
\begin{bmatrix}
\cos \psi & 0 & \sin \psi \
0 & 1 & 0 \
-\sin \psi & 0 & \cos \psi
\end{bmatrix}
]

Projection:

[
p_{2D} = K \cdot (R_y C + T)
]

---

## 6️⃣ Kalman Filter for Yaw

State:

[
x =
\begin{bmatrix}
\psi \
\dot{\psi}
\end{bmatrix}
]

Prediction:

[
x_{k|k-1} = A x_{k-1}
]

Where:

[
A =
\begin{bmatrix}
1 & 1 \
0 & 1
\end{bmatrix}
]

Measurement update:

[
x_{k|k} = x_{k|k-1} + K (z_k - H x_{k|k-1})
]

---

## 7️⃣ Position Kalman

State:

[
[x, y, \dot{x}, \dot{y}]
]

Used to stabilize bottom anchor of 3D box.

---

# 📦 Model Sources

---

## 🔹 YOLOv9

Paper:

> Wang et al., YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (2024)

Repo:
[https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)

---

## 🔹 DeepSORT

Paper:

> Wojke et al., Simple Online and Realtime Tracking (2017)

Repo:
[https://github.com/ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)

---

## 🔹 YOLOP

Paper:

> YOLOP: You Only Look Once for Panoptic Driving Perception (2021)

Repo:
[https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)

---

## 🔹 SAM 2

Meta AI:

> Segment Anything Model 2

Repo:
[https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)

Ultralytics integration:
[https://docs.ultralytics.com](https://docs.ultralytics.com)

---

## 🔹 Depth Anything V3

Paper:

> Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data (2024)

Model:
depth-anything/DA3METRIC-LARGE

Repo:
[https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
[https://github.com/DepthAnything/Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-3)

---

## 🔹 Intel DPT

Paper:

> Vision Transformers for Dense Prediction (DPT)

Repo:
[https://github.com/isl-org/DPT](https://github.com/isl-org/DPT)

---

## 🔹 Open3D

Library:
[http://www.open3d.org/](http://www.open3d.org/)

---

# 🚀 Installation

```bash
git clone https://github.com/yourname/AV-Multimodal-Perception-System
cd AV-Multimodal-Perception-System

pip install -r requirements.txt
```

If using Depth Anything V2:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

---

# ▶️ Run

```bash
python av_perception_system.py --source video.mp4
```

---

# 📤 Outputs

* BEV overlay video
* `bev_topdown.ply`
* `pointcloud_with_boxes.ply`
* `textured_pointcloud.ply`
* `depth_log.csv`

---

# 🔬 Research Contributions

* Rolling pseudo-LiDAR buffer
* Adaptive 3D bounding box scaling
* EMA + Kalman fusion stabilization
* BEV semantic recoloring
* Exportable 3D scene reconstruction

---

# 📈 Future Extensions

* Multi-frame SLAM fusion
* Ego-motion compensation
* Multi-camera fusion
* Radar/LiDAR sensor fusion
* Transformer-based 3D box estimation
* Town-scale mapping expansion

---

# 🏁 Conclusion

This repository implements a **full perception stack approximating production autonomous driving pipelines**, combining:

* Detection
* Tracking
* Segmentation
* Depth
* 3D reconstruction
* Stabilized 3D bounding boxes
* BEV pseudo-LiDAR mapping
* Exportable point clouds
