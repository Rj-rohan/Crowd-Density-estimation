"""
Generate Gaussian density maps from ShanghaiTech .mat annotations.

ShanghaiTech structure:
  ShanghaiTech/
    part_A/
      train_data/
        images/   processed_IMG_N.jpg
        ground-truth/  GT_IMG_N.mat
      test_data/
        images/
        ground-truth/
    part_B/  (same structure)

.mat annotation format:
  image_info[0][0][0][0][0]  ->  ndarray of shape (N, 2)  [x, y] head points

Usage:
  python data/generate_density_maps.py --root ShanghaiTech --part A --split train_data
  python data/generate_density_maps.py --root ShanghaiTech --part A --split test_data
  python data/generate_density_maps.py --root ShanghaiTech --part B --split train_data
  python data/generate_density_maps.py --root ShanghaiTech --part B --split test_data
"""

import os
import argparse
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_points(mat_path):
    mat = sio.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]  # shape (N, 2): [x, y]
    return points


def generate_density_map(img_shape, points, sigma=15):
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        density[y, x] += 1.0
    return gaussian_filter(density, sigma=sigma)


def process(root, part, split):
    base = os.path.join(root, f"part_{part}", split)
    img_dir = os.path.join(base, "images")
    gt_dir = os.path.join(base, "ground-truth")
    out_dir = os.path.join(base, "maps")
    os.makedirs(out_dir, exist_ok=True)

    mat_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".mat"))
    for mat_file in mat_files:
        # GT_IMG_N.mat  →  processed_IMG_N.jpg
        n = mat_file.replace("GT_IMG_", "").replace(".mat", "")
        img_path = os.path.join(img_dir, f"processed_IMG_{n}.jpg")
        mat_path = os.path.join(gt_dir, mat_file)

        if not os.path.exists(img_path):
            print(f"  Missing image: {img_path}")
            continue

        img = np.array(Image.open(img_path))
        points = load_points(mat_path)
        density = generate_density_map(img.shape, points)
        np.save(os.path.join(out_dir, f"IMG_{n}.npy"), density)

    print(f"  ✓ Generated {len(mat_files)} density maps → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ShanghaiTech")
    parser.add_argument("--part", default="A", choices=["A", "B"])
    parser.add_argument("--split", default="train_data", choices=["train_data", "test_data"])
    args = parser.parse_args()
    process(args.root, args.part, args.split)
