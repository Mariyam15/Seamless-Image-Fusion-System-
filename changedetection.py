# BSAI23013
# MARIYAM FATIMA
# ASSIGNMENT 1

import os
import math
import numpy as np
from PIL import Image  # only for reading
import matplotlib.pyplot as plt
import imageio


# ------------------- Task 1: Read Images ------------------- #
def read_images(folder: str, ext: str = "png") -> np.ndarray:
    """
    Reads grayscale images from a folder and returns them as a NumPy array.
    """
    data = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(ext.lower()):
            gray = Image.open(os.path.join(folder, f)).convert("L")  # convert to grayscale
            data.append(np.array(gray, dtype=np.float32))
    return np.array(data)


# ------------------- Task 2: Plot Frames ------------------- #
def plot_frames(frames: np.ndarray, num_frames: int, save_name: str) -> None:
    """
    Plot frames in a grid with 5 frames per row, show inline, and save as a PDF.
    """
    rows = math.ceil(num_frames / 5)
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel()

    for i in range(num_frames):
        axes[i].imshow(frames[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Frame {i}")

    # Hide unused subplots
    for j in range(num_frames, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)


# ------------------- Task 3: Mean & Variance ------------------- #
def compute_mean(frames: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel mean across frames.
    """
    N = frames.shape[0]
    mean_frame = np.sum(frames, axis=0) / N
    return mean_frame.astype(np.float32)


def compute_variance(frames: np.ndarray, mean_frame: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel variance across frames.
    """
    N = frames.shape[0]
    variance_frame = np.sum((frames - mean_frame) ** 2, axis=0) / N
    return variance_frame.astype(np.float32)


# ------------------- Task 4: Change Detection ------------------- #
def compute_mask(frame: np.ndarray, mean_frame: np.ndarray, variance_frame: np.ndarray,
                 threshold: float = 5.0) -> np.ndarray:
    """
    Compute binary mask using Mahalanobis distance.
    If distance > threshold → foreground (255), else background (0).
    """
    var_safe = np.where(variance_frame == 0, 1e-6, variance_frame)
    dist = (frame - mean_frame) ** 2 / var_safe
    mask = (dist > threshold).astype(np.uint8) * 255
    return mask


# ------------------- Task 5: Morphological Ops ------------------- #
def dilation(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    ksize = kernel_size
    pad = ksize // 2
    padded = np.pad(binary_img, pad, mode="constant", constant_values=0)
    dilated = np.zeros_like(binary_img)

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            region = padded[i:i + ksize, j:j + ksize]
            if np.any(region == 255):
                dilated[i, j] = 255
    return dilated


def erosion(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    ksize = kernel_size
    pad = ksize // 2
    padded = np.pad(binary_img, pad, mode="constant", constant_values=0)
    eroded = np.zeros_like(binary_img)

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            region = padded[i:i + ksize, j:j + ksize]
            if np.all(region == 255):
                eroded[i, j] = 255
    return eroded


def morphological_operations(mask, kernel_size=3):
    """
    Apply opening: erosion followed by dilation.
    """
    eroded = erosion(mask, kernel_size)
    dilated = dilation(eroded, kernel_size)
    return dilated


# ------------------- Task 6: Connected Components ------------------- #
def connected_components(binary_img: np.ndarray):
    """
    Find connected components in a binary image using flood fill.
    Returns: num_labels, labels, stats (area, centroid, bbox)
    """
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=int)
    current_label = 0
    stats = []

    def flood_fill(i, j, lbl):
        stack = [(i, j)]
        pixels = []
        while stack:
            x, y = stack.pop()
            if (0 <= x < h and 0 <= y < w and
                binary_img[x, y] == 255 and labels[x, y] == 0):
                labels[x, y] = lbl
                pixels.append((x, y))
                stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
        if not pixels:
            return None
        xs, ys = zip(*pixels)
        area = len(pixels)
        centroid = (np.mean(xs), np.mean(ys))
        bbox = (min(xs), min(ys), max(xs), max(ys))
        return {"label": lbl, "area": area, "centroid": centroid, "bbox": bbox}

    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255 and labels[i, j] == 0:
                current_label += 1
                blob_stats = flood_fill(i, j, current_label)
                if blob_stats:
                    stats.append(blob_stats)

    return current_label, labels, stats


# ------------------- Task 7: Alpha Blending ------------------- #
def alpha_blend(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend foreground with background in masked regions using alpha (no cv2).
    """
    mask = (mask > 0).astype(np.float32)
    if foreground.ndim == 3 and mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    blended = fg * (1 - alpha * mask) + bg * (alpha * mask)
    return blended.astype(np.uint8)


# ------------------- Task 8: Save Masks & Video ------------------- #
def remove_person_alpha(frames: np.ndarray, mean_img: np.ndarray, var_img: np.ndarray,
                        masks_folder: str = "output_masks", output_folder: str = "output_faded",
                        video_path: str = "faded_person.mp4", threshold: float = 2.0,
                        kernel_size: int = 5, fps: int = 10):
    """
    Remove person gradually using alpha blending.
    Saves masks, blended frames, and final video.
    """
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    blended_frames = []
    N = len(frames)

    for idx, frame in enumerate(frames):
        # Step 4: compute mask
        mask = compute_mask(frame, mean_img, var_img, threshold=threshold)
        mask = morphological_operations(mask, kernel_size=kernel_size)

        # Save mask
        mask_path = os.path.join(masks_folder, f"mask_{idx:04d}.png")
        imageio.imwrite(mask_path, mask.astype(np.uint8))

        # Expand frames to 3 channels
        mask_3c = np.repeat((mask > 0)[:, :, None], 3, axis=2)
        k_frame = np.repeat(frame[:, :, None], 3, axis=2)
        t_frame = np.repeat(mean_img[:, :, None], 3, axis=2)

        # Alpha increases gradually
        alpha = idx / (N - 1)

        # Alpha blending
        blended = k_frame * (1 - alpha * mask_3c) + t_frame * (alpha * mask_3c)
        blended = blended.astype(np.uint8)

        # Save blended frame
        out_path = os.path.join(output_folder, f"frame_{idx:04d}.png")
        imageio.imwrite(out_path, blended)

        blended_frames.append(blended)

    # Save video
    imageio.mimsave(video_path, blended_frames, fps=fps, codec="libx264")
    print(f"Masks in '{masks_folder}', frames in '{output_folder}', video at '{video_path}'")
