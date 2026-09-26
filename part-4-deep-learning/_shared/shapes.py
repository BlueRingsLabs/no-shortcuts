"""Synthetic shape images for the vision labs (Module 27 and later).

Why synthetic? The labs must run offline, on a CPU, in a couple of minutes, and I want the ground truth for boxes and
masks to be exact. Real datasets (CIFAR, COCO, Pascal VOC) are for the capstones and for your own curiosity.

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
    from shapes import make_classification, make_scenes, CLASSES

make_classification(n, size=32, seed=0, clutter=2, r_range=(0.12, 0.3)) -> images (n, 3, size, size) float32 in [0, 1], labels (n,)
    one shape per image, random size, position and color, on a noisy gradient background with clutter lines.

make_colored(n, size=32, seed=0, clutter=2) -> images, shape labels, color labels
    like make_classification, but the object has one of the named COLORS (for captions: "a red circle").

make_scenes(n, size=64, max_objects=4, seed=0) -> images, targets
    several shapes per image (they may overlap; later ones are drawn on top). targets[i] is a dict with
    "boxes" (k, 4) as x1, y1, x2, y2 in pixels (inclusive-exclusive), "labels" (k,), "masks" (k, size, size) bool
    visible-pixel instance masks, and "semantic" (size, size) int with 0 = background and class + 1 elsewhere.
"""

from __future__ import annotations

import numpy as np

CLASSES = ["circle", "square", "triangle", "plus"]
COLORS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.1), "blue": (0.15, 0.25, 0.95),
          "yellow": (0.95, 0.9, 0.1), "magenta": (0.9, 0.1, 0.9), "cyan": (0.1, 0.9, 0.9)}


def _shape_mask(kind: str, cx: float, cy: float, r: float, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    dx, dy = xx - cx, yy - cy
    if kind == "circle":
        return dx**2 + dy**2 <= r**2
    if kind == "square":
        return (np.abs(dx) <= r * 0.85) & (np.abs(dy) <= r * 0.85)
    if kind == "triangle":                                      # apex up
        t = (dy + r) / (2 * r)                                   # 0 at the apex, 1 at the base
        return (dy >= -r) & (dy <= r) & (np.abs(dx) <= t * r)
    if kind == "plus":
        w = max(r * 0.35, 1.0)
        return ((np.abs(dx) <= w) & (np.abs(dy) <= r)) | ((np.abs(dy) <= w) & (np.abs(dx) <= r))
    raise ValueError(kind)


def _background(rng: np.random.Generator, size: int, clutter: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = rng.uniform(0.1, 0.5, 3)[:, None, None]
    grad = rng.uniform(-0.2, 0.2, (3, 1, 1)) * xx[None] + rng.uniform(-0.2, 0.2, (3, 1, 1)) * yy[None]
    img = base + grad + rng.normal(0, 0.04, (3, size, size))
    for _ in range(clutter):                                    # thin random lines: distractors, not objects
        x0, y0, x1, y1 = rng.uniform(0, size, 4)
        t = np.linspace(0, 1, size * 2)
        px = np.clip((x0 + t * (x1 - x0)).astype(int), 0, size - 1)
        py = np.clip((y0 + t * (y1 - y0)).astype(int), 0, size - 1)
        img[:, py, px] = rng.uniform(0, 1, 3)[:, None]
    return img


def _color(rng: np.random.Generator) -> np.ndarray:
    c = rng.uniform(0, 1, 3)
    c[rng.integers(3)] = rng.uniform(0.75, 1.0)                  # at least one bright channel: visible on the background
    return c


def make_classification(n: int, size: int = 32, seed: int = 0, clutter: int = 2, r_range: tuple = (0.12, 0.3)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    images = np.empty((n, 3, size, size), dtype=np.float32)
    labels = rng.integers(0, len(CLASSES), n)
    for i in range(n):
        img = _background(rng, size, clutter)
        r = rng.uniform(size * r_range[0], size * r_range[1])
        cx, cy = rng.uniform(r, size - r, 2)
        m = _shape_mask(CLASSES[labels[i]], cx, cy, r, yy, xx)
        img[:, m] = _color(rng)[:, None]
        images[i] = np.clip(img, 0, 1)
    return images, labels


def make_colored(n: int, size: int = 32, seed: int = 0, clutter: int = 2, r_range: tuple = (0.15, 0.3)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    names = list(COLORS)
    images = np.empty((n, 3, size, size), dtype=np.float32)
    shapes, colors = rng.integers(0, len(CLASSES), n), rng.integers(0, len(names), n)
    for i in range(n):
        img = _background(rng, size, clutter) * 0.6                   # a darker background, so the named colors stand out
        r = rng.uniform(size * r_range[0], size * r_range[1])
        cx, cy = rng.uniform(r, size - r, 2)
        m = _shape_mask(CLASSES[shapes[i]], cx, cy, r, yy, xx)
        img[:, m] = (np.array(COLORS[names[colors[i]]]) + rng.normal(0, 0.05, 3))[:, None]
        images[i] = np.clip(img, 0, 1)
    return images, shapes, colors


def make_scenes(n: int, size: int = 64, max_objects: int = 4, seed: int = 0, clutter: int = 3, min_visible: int = 12):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    images = np.empty((n, 3, size, size), dtype=np.float32)
    targets = []
    for i in range(n):
        img = _background(rng, size, clutter)
        k = rng.integers(1, max_objects + 1)
        owner = np.full((size, size), -1)
        labels, raw_masks = [], []
        for j in range(k):
            lab = rng.integers(0, len(CLASSES))
            r = rng.uniform(size * 0.07, size * 0.2)
            cx, cy = rng.uniform(r, size - r, 2)
            m = _shape_mask(CLASSES[lab], cx, cy, r, yy, xx)
            img[:, m] = _color(rng)[:, None]
            owner[m] = j
            labels.append(lab); raw_masks.append(m)
        boxes, keep_labels, masks = [], [], []
        semantic = np.zeros((size, size), dtype=np.int64)
        for j, lab in enumerate(labels):
            vis = owner == j                                     # pixels of this object still visible after later ones
            if vis.sum() < min_visible:
                continue
            ys, xs = np.nonzero(vis)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            keep_labels.append(lab); masks.append(vis)
            semantic[vis] = lab + 1
        images[i] = np.clip(img, 0, 1)
        targets.append({"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4), "labels": np.array(keep_labels, dtype=np.int64),
                        "masks": np.array(masks, dtype=bool).reshape(-1, size, size), "semantic": semantic})
    return images, targets
