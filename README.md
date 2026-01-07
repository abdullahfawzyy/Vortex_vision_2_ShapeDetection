# Geometric Shape Detection

## Introduction
This module implements a computer vision algorithm to detect, classify, and count geometric shapes (Triangles, Squares, Rectangles, and Circles) from an input image. 

## Features
- **Noise Filtering:** Ignores small artifacts and background noise using area thresholding.
- **Border Rejection:** Automatically filters out the page boundaries or large background elements.
- **Shape Classification:** Distinguishes between:
  - **Triangles** (3 vertices)
  - **Squares** (4 vertices, aspect ratio ≈ 1.0)
  - **Rectangles** (4 vertices, aspect ratio ≠ 1.0)
  - **Circles** (> 4 vertices)
- **Visual Output:** Generates a processed image with contours drawn and shapes labeled.

## How It Works
The algorithm follows a standard Computer Vision pipeline using OpenCV:

1.  **Preprocessing:**
    - The image is converted to **Grayscale** to rely on intensity rather than color.
    - A **Gaussian Blur** (5x5 kernel) is applied to smooth edges and reduce high-frequency noise.
    - **Inverse Binary Thresholding** is used to separate the dark shapes from the light background.

2.  **Contour Detection:**
    - The script finds all contours (boundaries) in the binary image.

3.  **Filtering & Calibration:**
    - **Area Filter:** Contours smaller than 500px (noise) or larger than 50,000px (borders) are ignored.
    - **Approximation:** The Douglas-Peucker algorithm (`approxPolyDP`) simplifies the contour shape.
    - *Calibration Note:* The tolerance was tuned to **2% of the perimeter**. This prevents circles from being over-simplified into triangles while still keeping squares sharp.

4.  **Classification Logic:**
    - The number of vertices in the approximated polygon determines the shape:
        - **3 Vertices:** Triangle.
        - **4 Vertices:** Square or Rectangle (checked via Aspect Ratio).
        - **>4 Vertices:** Circle.

## Setup & Usage

### Prerequisites
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

### Installation
```bash
pip install opencv-python numpy
