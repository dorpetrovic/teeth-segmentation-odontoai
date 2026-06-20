"""
utils/visualization.py 

Plotting functions for dental panoramic X-ray images.
Supports FDI multi-class notation (tooth-11 … tooth-85): adult + deciduous teeth.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import random
import math
from pathlib import Path

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import utils.preprocessing
import importlib
importlib.reload(utils.preprocessing)

from utils.preprocessing import (
    count_teeth_per_image,
    class_frequency,
    load_image,
    build_masks,
    enhance_contrast,
    images_missing_annotations,
    split_summary,
)


# ── Colour scheme ─────────────────────────────────────────────────────────────

QUADRANT_COLORS = {
    "UR": "#4A90D9",   # upper right permanent  — blue
    "UL": "#E87040",   # upper left  permanent  — orange
    "LL": "#2ECC71",   # lower left  permanent  — green
    "LR": "#9B59B6",   # lower right permanent  — purple
    "DU": "#F1C40F",   # deciduous upper        — yellow
    "DL": "#E74C3C",   # deciduous lower        — red
}

LEGEND_PATCHES = [
    patches.Patch(facecolor=QUADRANT_COLORS["UR"], label="UR 11-18"),
    patches.Patch(facecolor=QUADRANT_COLORS["UL"], label="UL 21-28"),
    patches.Patch(facecolor=QUADRANT_COLORS["LL"], label="LL 31-38"),
    patches.Patch(facecolor=QUADRANT_COLORS["LR"], label="LR 41-48"),
    patches.Patch(facecolor=QUADRANT_COLORS["DU"], label="Deciduous U 51-65"),
    patches.Patch(facecolor=QUADRANT_COLORS["DL"], label="Deciduous L 71-85"),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _quadrant_color(class_name):
    if class_name == 'tooth':
        return "#4A90D9"
    try:
        fdi = int(class_name.split('-')[1])  # 'tooth-11' → 11
        if 11 <= fdi <= 18: return QUADRANT_COLORS["UR"]
        if 21 <= fdi <= 28: return QUADRANT_COLORS["UL"]
        if 31 <= fdi <= 38: return QUADRANT_COLORS["LL"]
        if 41 <= fdi <= 48: return QUADRANT_COLORS["LR"]
        if 51 <= fdi <= 65: return QUADRANT_COLORS["DU"]
        if 71 <= fdi <= 85: return QUADRANT_COLORS["DL"]
    except (ValueError, IndexError):
        pass
    return "#AAAAAA"


def apply_masks(image, masks, class_names = None, alpha = 0.45):
    
    """
    Draw semi-transparent colored masks on X-ray image.

    Args:
        image: RGB uint8 (H, W, 3).
        masks: Bool array (H, W, N) — one per tooth.
        class_names: Category name per mask for color 
        alpha: 0 = invisible, 1 = solid

    Returns:
        Annotated RGB uint8 image.
    """

    output = image.copy().astype(np.float32)
    n = masks.shape[-1]

    for i in range(n):
        if class_names and i < len(class_names):
            hex_col = _quadrant_color(class_names[i])
        else:
            cmap = cm.get_cmap("tab20", max(1, n))
            hex_col = "#{:02x}{:02x}{:02x}".format(
                *[int(v * 255) for v in cmap(i)[:3]])

        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        colour = np.array([r, g, b], dtype=np.float32)

        for c in range(3):
            output[:, :, c] = np.where(
                masks[:, :, i],
                output[:, :, c] * (1 - alpha) + colour[c] * alpha,
                output[:, :, c],
            )
    return output.astype(np.uint8)


def draw_bounding_boxes(image, rois, class_ids, scores, class_names):

    """
    Draw bounding boxes with FDI label and confidence score on image.

    Args:
        image: (H, W, 3) uint8.
        rois: (N, 4) — [y1, x1, y2, x2] per detection.
        class_ids: (N,)  — index into class_names.
        scores: (N,)  — confidence in [0, 1].
        class_names: List including background at index 0.
    """

    out = image.copy()
    for i, roi in enumerate(rois):
        y1, x1, y2, x2 = roi
        name = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "unknown"
        hex_col = _quadrant_color(name)
        color = (int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16))
        label = f"{name} {scores[i]:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def visualize_prediction(image, result, class_names):
    
    """
    Side-by-side view of the original X-ray and the model's predictions.
    Draws both masks and b.boxes on the right panel.

    Args:
        image: Original X-ray (H, W, 3) uint8.
        result: Dict with keys: masks, rois, class_ids, scores.
        class_names: Class list — index 0 is background.
    """

    masks = result.get("masks", np.zeros((*image.shape[:2], 0), dtype=bool))
    rois = result.get("rois", np.zeros((0, 4), dtype=int))
    class_ids = result.get("class_ids", np.array([], dtype=int))
    scores = result.get("scores", np.array([], dtype=float))

    det_names = [class_names[cid] for cid in class_ids if cid < len(class_names)]
    annotated = apply_masks(image, masks, det_names)
    annotated = draw_bounding_boxes(annotated, rois, class_ids, scores, class_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")
    axes[1].imshow(annotated, cmap="gray")
    axes[1].set_title(f"{masks.shape[-1]} teeth detected")
    axes[1].axis("off")
    axes[1].legend(handles=LEGEND_PATCHES, loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return annotated


def plot_annotation_distribution(train_coco, val_coco):
    
    """
    Two histograms side by side:
      Left -> train annotation count distribution with mean & median lines.
      Right -> train vs val overlay to confirm both splits look similar.
    """

    train_vals = list(count_teeth_per_image(train_coco).values())
    val_vals = list(count_teeth_per_image(val_coco).values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(train_vals, bins=25, color="#185FA5", edgecolor="white")
    axes[0].axvline(np.mean(train_vals),   color="#D85A30", linestyle="--", linewidth=1.5,
                    label=f"Mean = {np.mean(train_vals):.1f}")
    axes[0].axvline(np.median(train_vals), color="#2ECC71", linestyle="--", linewidth=1.5,
                    label=f"Median = {np.median(train_vals):.1f}")
    axes[0].set_xlabel("Instances per image"); axes[0].set_ylabel("Images")
    axes[0].set_title(f"Train annotation distribution ({len(train_coco['images'])} images)")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(train_vals, bins=25, color="#185FA5", edgecolor="white", alpha=0.7, label="Train")
    axes[1].hist(val_vals,   bins=25, color="#E87040", edgecolor="white", alpha=0.7, label="Val")
    axes[1].set_xlabel("Instances per image"); axes[1].set_ylabel("Images")
    axes[1].set_title("Train vs Val distribution")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_category_frequency(coco):

    """
    2×3 grid of bar charts, one panel per anatomical group
    Makes it easy to spot under-represented teeth within each quadrant
    """

    freq = class_frequency(coco) #class_id: count of teeth (across all annotations)

    fdi_counts = {}
    for cat in coco["categories"]:
        try:
            fdi = int(cat["name"].split('-')[1])
            fdi_counts[fdi] = freq.get(cat["id"], 0)
        except (ValueError, IndexError):
            pass

    groups = {
        "UR (11-18)": (list(range(11, 19)), QUADRANT_COLORS["UR"]),
        "UL (21-28)": (list(range(21, 29)), QUADRANT_COLORS["UL"]),
        "LL (31-38)": (list(range(31, 39)), QUADRANT_COLORS["LL"]),
        "LR (41-48)": (list(range(41, 49)), QUADRANT_COLORS["LR"]),
        "Deciduous U (51-65)": (list(range(51, 56)) + list(range(61, 66)), QUADRANT_COLORS["DU"]),
        "Deciduous L (71-85)": (list(range(71, 76)) + list(range(81, 86)), QUADRANT_COLORS["DL"]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle(f"Annotation Frequency by FDI Group (train, {len(coco['images'])} images)", fontsize=13)

    for ax, (label, (fdi_range, color)) in zip(axes, groups.items()):
        counts = [fdi_counts.get(fdi, 0) for fdi in fdi_range]
        ax.bar(range(len(fdi_range)), counts, color=color, edgecolor="white")
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(fdi_range)))
        ax.set_xticklabels([str(f) for f in fdi_range], rotation=45, fontsize=8)
        ax.set_xlabel("FDI number"); ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Annotation count")
    axes[3].set_ylabel("Annotation count")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    all_counts = list(fdi_counts.values())
    print(f"Most common: tooth-{max(fdi_counts, key=fdi_counts.get)}  ({max(all_counts)} annotations)")
    print(f"Least common: tooth-{min(fdi_counts, key=fdi_counts.get)}  ({min(all_counts)} annotations)")
    print(f"Imbalance ratio: {max(all_counts) / max(min(all_counts), 1):.1f}x")


def plot_sample_annotations(coco, img_dir, n = 4, seed = 42):
   
    """
    n randomly sampled images with polygon colored by FDI quadrant.
    Only samples images that have annotations.
    """
    random.seed(seed)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    imgs_with_anns = [img for img in coco["images"] if img["id"] in anns_by_image]
    sample_imgs = random.sample(imgs_with_anns, min(n, len(imgs_with_anns)))

    fig, axes = plt.subplots(1, len(sample_imgs), figsize=(5 * len(sample_imgs), 5))
    if len(sample_imgs) == 1:
        axes = [axes]
    fig.suptitle("Sample Annotations — polygon overlays by FDI quadrant", fontsize=12)

    for ax, img_info in zip(axes, sample_imgs):
        from pathlib import Path
        img  = load_image(str(Path(img_dir) / img_info["file_name"]))
        anns = anns_by_image.get(img_info["id"], [])
        ax.imshow(img, cmap="gray")
        for ann in anns:
            seg = ann["segmentation"][0]
            xs = seg[0::2] + [seg[0]]
            ys = seg[1::2] + [seg[1]]
            color = _quadrant_color(cat_map.get(ann["category_id"], ""))
            ax.plot(xs, ys, "-", color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(f"{img_info['file_name']}\n({len(anns)} instances)", fontsize=9)
        ax.axis("off")

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
    plt.close(fig)


def plot_mask_verification(coco, img_dir):
    
    """
    Render pixel-level masks for one image — confirms polygon → mask
    conversion works correctly end to end.
    """

    annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
    img_info = next(img for img in coco["images"] if img["id"] in annotated_ids)
    img = load_image(str(Path(img_dir) / img_info["file_name"]))
    H, W = img.shape[:2]

    masks, class_ids = build_masks(coco, img_info["file_name"], H, W)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    class_names = [cat_map.get(cid, "unknown") for cid in class_ids]
    
    annotated = apply_masks(img, masks, class_names)

    print(f"Image:  {img_info['file_name']}  ({W}×{H}px)")
    print(f"Masks:  {masks.shape[2]}  |  any filled: {masks.any()}")
    print(f"Sample classes: {class_names[:6]}...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].imshow(img, cmap="gray"); 
    axes[0].set_title("Original");
    axes[0].axis("off")
    axes[1].imshow(annotated, cmap="gray");
    axes[1].set_title(f"Masks ({masks.shape[2]} teeth)"); 
    axes[1].axis("off")
    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
    plt.close(fig)


def plot_contrast_comparison(coco, img_dir):
   
    """
    Show Original | CLAHE | Histogram EQ for 3  images
    """

    sample_files = [
        coco["images"][0]["file_name"],
        coco["images"][len(coco["images"]) // 2]["file_name"],
        coco["images"][-1]["file_name"],
    ]

    fig, axes = plt.subplots(len(sample_files), 3, figsize=(18, 4 * len(sample_files)))
    fig.suptitle("Contrast Enhancement: Original | CLAHE | Histogram EQ", fontsize=13)

    for row, fname in enumerate(sample_files):
        orig = load_image(str(Path(img_dir) / fname))
        clahe = enhance_contrast(orig, method="clahe")
        heq = enhance_contrast(orig, method="histogram_eq")
        for col, (img, title) in enumerate([
            (orig, f"{fname} — Original"),
            (clahe, "CLAHE"),
            (heq, "Histogram EQ"),
        ]):
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_tooth_sizes(coco, mean_image_height, target_short_side = 800):
    
    """
    Histogram of tooth width, height, and diagonal in molded image space.
    Also prints recommended anchor scales derived from the data.
    """

    scale = target_short_side / mean_image_height
    ws = [ann["bbox"][2] * scale for ann in coco["annotations"]]
    hs = [ann["bbox"][3] * scale for ann in coco["annotations"]]
    diags = [np.sqrt(w**2 + h**2) for w, h in zip(ws, hs)]

    print(f"Resize scale: {scale:.3f}")
    print(f"Width: mean={np.mean(ws):.0f}  median={np.median(ws):.0f}px")
    print(f"Height: mean={np.mean(hs):.0f}  median={np.median(hs):.0f}px")
    print(f"Diagonal: min={min(diags):.0f}  mean={np.mean(diags):.0f}  max={max(diags):.0f}px")

    s = max(4, 2 ** (math.floor(math.log2(min(ws + hs))) - 1))
    scales = []
    while s < max(diags) * 1.5:
        scales.append(int(s)); s *= 2
    print(f"Recommended anchor scales: {tuple(scales)}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title, color in zip(
        axes,
        [ws, hs, diags],
        ["Tooth width (px)", "Tooth height (px)", "Tooth diagonal (px)"],
        ["#4A90D9", "#E87040", "#2ECC71"]
    ):
        ax.hist(data, bins=25, color=color, edgecolor="white")
        ax.set_title(title); ax.set_xlabel("pixels"); ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Tooth sizes in molded space (short side={target_short_side}px)")
    plt.tight_layout()
    plt.show()
    plt.close(fig)