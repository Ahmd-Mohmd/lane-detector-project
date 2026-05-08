from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st

try:
    from PIL import Image
except Exception:  # pragma: no cover - pillow not installed
    Image = None

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional for video
    imageio_ffmpeg = None

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except Exception:  # pragma: no cover - optional for video
    VideoFileClip = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global parameters
KERNEL_SIZE = 5
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150
TRAP_BOTTOM_WIDTH = 0.85
TRAP_TOP_WIDTH = 0.08
TRAP_HEIGHT = 0.4
HOUGH_RHO = 2
HOUGH_THETA = 1 * np.pi / 180
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 20


VIDEO_OPTIONS = {
    "Option 1 - Full quality (slowest)": {"scale": 1.0, "fps": None},
    "Option 2 - Faster: reduced FPS only": {"scale": 1.0, "fps": 15},
    "Option 3 - Faster: half resolution": {"scale": 0.5, "fps": None},
    "Option 4 - Fastest: half res + low FPS": {"scale": 0.5, "fps": 15},
}


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Space+Grotesk:wght@400;600&display=swap');

:root {
    --bg-1: #f5f7fb;
    --bg-2: #ffffff;
    --bg-3: #eef2f8;
    --card: rgba(255, 255, 255, 0.9);
    --card-border: rgba(15, 23, 42, 0.1);
    --accent: #1d4ed8;
    --accent-2: #f59e0b;
    --text-main: #0f172a;
    --text-muted: #526079;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 600px at 10% 8%, #e2e8f0 0%, var(--bg-1) 40%, var(--bg-2) 100%);
    color: var(--text-main);
}

[data-testid="stHeader"] {
    background: transparent;
}

* {
  font-family: 'Space Grotesk', sans-serif;
}

h1, h2, h3 {
  font-family: 'DM Serif Display', serif;
  letter-spacing: 0.5px;
}

.hero {
    padding: 24px 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(29, 78, 216, 0.12), rgba(245, 158, 11, 0.12));
    border: 1px solid var(--card-border);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.panel {
    background: var(--card);
    border: 1px solid var(--card-border);
    padding: 16px 18px;
    border-radius: 14px;
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
}

.panel + .panel {
  margin-top: 16px;
}

small, .muted {
  color: var(--text-muted);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
    border-right: 1px solid rgba(15, 23, 42, 0.08);
}

.stButton button {
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    color: #ffffff;
    border: none;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}

.stButton button:hover {
  filter: brightness(1.05);
}

@media (max-width: 900px) {
  .hero { padding: 18px; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype.kind == "f":
        if array.max() <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _as_uint8_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype.kind == "f":
        if array.max() <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _save_image_bytes(image: np.ndarray, output_path: Path) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow is required for image export.")
    array = np.asarray(image)
    if array.ndim == 2:
        pil_image = Image.fromarray(_as_uint8_gray(array), mode="L")
    else:
        pil_image = Image.fromarray(_as_uint8_rgb(array))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    output_path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def load_image_from_upload(uploaded_file: Any) -> Tuple[np.ndarray, bytes]:
    data = uploaded_file.getvalue()
    if Image is None:
        raise RuntimeError("Pillow is required to load images.")
    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(pil_image), data


def grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    rgb = array[..., :3].astype(np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def filter_colors(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image).astype(np.uint8)
    rgb = array[..., :3]
    white_mask = (
        (rgb[..., 0] >= 200)
        & (rgb[..., 1] >= 200)
        & (rgb[..., 2] >= 200)
    )
    yellow_mask = (
        (rgb[..., 0] >= 160)
        & (rgb[..., 1] >= 140)
        & (rgb[..., 2] <= 140)
        & (rgb[..., 0] >= rgb[..., 1])
        & (rgb[..., 1] >= rgb[..., 2])
    )
    filtered = np.zeros_like(rgb)
    selected = white_mask | yellow_mask
    filtered[selected] = rgb[selected]
    return filtered


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: Optional[float] = None) -> np.ndarray:
    array = np.asarray(image).astype(np.float64)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma is None:
        sigma = kernel_size / 6.0
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    if array.ndim == 2:
        padded = np.pad(array, radius, mode="reflect")
        blurred = np.zeros_like(array)
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                patch = padded[row : row + kernel_size, col : col + kernel_size]
                blurred[row, col] = np.sum(patch * kernel)
        return blurred

    blurred_channels = []
    for channel in range(array.shape[2]):
        padded = np.pad(array[..., channel], radius, mode="reflect")
        blurred_channel = np.zeros(array.shape[:2], dtype=np.float64)
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                patch = padded[row : row + kernel_size, col : col + kernel_size]
                blurred_channel[row, col] = np.sum(patch * kernel)
        blurred_channels.append(blurred_channel)
    return np.stack(blurred_channels, axis=-1)


def canny(image: np.ndarray, low_threshold: int = LOW_THRESHOLD, high_threshold: int = HIGH_THRESHOLD) -> np.ndarray:
    array = np.asarray(image).astype(np.float64)
    if array.ndim != 2:
        raise ValueError("canny expects a single-channel grayscale image")

    sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float64,
    )
    sobel_y = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=np.float64,
    )

    padded = np.pad(array, 1, mode="reflect")
    gradient_x = np.zeros_like(array)
    gradient_y = np.zeros_like(array)

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            patch = padded[row : row + 3, col : col + 3]
            gradient_x[row, col] = np.sum(patch * sobel_x)
            gradient_y[row, col] = np.sum(patch * sobel_y)

    magnitude = np.hypot(gradient_x, gradient_y)
    direction = np.degrees(np.arctan2(gradient_y, gradient_x))
    direction = (direction + 180.0) % 180.0

    suppressed = np.zeros_like(magnitude)
    for row in range(1, array.shape[0] - 1):
        for col in range(1, array.shape[1] - 1):
            angle = direction[row, col]
            current = magnitude[row, col]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = (magnitude[row, col - 1], magnitude[row, col + 1])
            elif 22.5 <= angle < 67.5:
                neighbors = (magnitude[row - 1, col + 1], magnitude[row + 1, col - 1])
            elif 67.5 <= angle < 112.5:
                neighbors = (magnitude[row - 1, col], magnitude[row + 1, col])
            else:
                neighbors = (magnitude[row - 1, col - 1], magnitude[row + 1, col + 1])

            if current >= neighbors[0] and current >= neighbors[1]:
                suppressed[row, col] = current

    strong_value = 255
    weak_value = 75
    result = np.zeros_like(suppressed, dtype=np.uint8)
    strong_rows, strong_cols = np.where(suppressed >= high_threshold)
    weak_rows, weak_cols = np.where((suppressed >= low_threshold) & (suppressed < high_threshold))
    result[strong_rows, strong_cols] = strong_value
    result[weak_rows, weak_cols] = weak_value

    stack = list(zip(strong_rows.tolist(), strong_cols.tolist()))
    while stack:
        row, col = stack.pop()
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                neighbor_row = row + delta_row
                neighbor_col = col + delta_col
                if 0 <= neighbor_row < result.shape[0] and 0 <= neighbor_col < result.shape[1]:
                    if result[neighbor_row, neighbor_col] == weak_value:
                        result[neighbor_row, neighbor_col] = strong_value
                        stack.append((neighbor_row, neighbor_col))

    result[result != strong_value] = 0
    return result


def region_of_interest(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    polygon = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    masked = np.zeros_like(array)

    min_x = max(int(np.floor(np.min(polygon[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(polygon[:, 0]))), array.shape[1] - 1)
    min_y = max(int(np.floor(np.min(polygon[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(polygon[:, 1]))), array.shape[0] - 1)

    for row in range(min_y, max_y + 1):
        for col in range(min_x, max_x + 1):
            inside = False
            j = len(polygon) - 1
            for i in range(len(polygon)):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                intersects = ((yi > row) != (yj > row)) and (
                    col < (xj - xi) * (row - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi
                )
                if intersects:
                    inside = not inside
                j = i

            if inside:
                if array.ndim == 2:
                    masked[row, col] = array[row, col]
                else:
                    masked[row, col, :] = array[row, col, :]

    return masked


def hough_lines(
    image: np.ndarray,
    rho: int = HOUGH_RHO,
    theta: float = HOUGH_THETA,
    threshold: int = HOUGH_THRESHOLD,
    min_line_len: int = HOUGH_MIN_LINE_LENGTH,
    max_line_gap: int = HOUGH_MAX_LINE_GAP,
) -> np.ndarray:
    edge_map = np.asarray(image)
    if edge_map.ndim != 2:
        raise ValueError("hough_lines expects a single-channel edge map")

    height, width = edge_map.shape
    ys, xs = np.nonzero(edge_map)
    if len(xs) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)

    rho_max = int(np.ceil(np.hypot(height, width)))
    rho_values = np.arange(-rho_max, rho_max + rho, rho)
    theta_values = np.arange(-np.pi / 2, np.pi / 2, theta)
    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=np.int32)

    cos_values = np.cos(theta_values)
    sin_values = np.sin(theta_values)

    for x, y in zip(xs, ys):
        rho_projection = x * cos_values + y * sin_values
        rho_indices = np.round((rho_projection + rho_max) / rho).astype(int)
        valid = (rho_indices >= 0) & (rho_indices < len(rho_values))
        accumulator[rho_indices[valid], np.arange(len(theta_values))[valid]] += 1

    peaks = []
    for rho_index in range(1, accumulator.shape[0] - 1):
        for theta_index in range(1, accumulator.shape[1] - 1):
            vote = accumulator[rho_index, theta_index]
            if vote < threshold:
                continue
            neighborhood = accumulator[rho_index - 1 : rho_index + 2, theta_index - 1 : theta_index + 2]
            if vote == np.max(neighborhood):
                peaks.append((rho_values[rho_index], theta_values[theta_index], vote))

    segments = []
    for rho_value, theta_value, _ in peaks:
        cos_theta = np.cos(theta_value)
        sin_theta = np.sin(theta_value)
        direction = np.array([-sin_theta, cos_theta], dtype=np.float64)

        distances = np.abs(xs * cos_theta + ys * sin_theta - rho_value)
        support = distances <= rho / 2.0
        if not np.any(support):
            continue

        support_x = xs[support].astype(np.float64)
        support_y = ys[support].astype(np.float64)
        projections = support_x * direction[0] + support_y * direction[1]

        order = np.argsort(projections)
        support_x = support_x[order]
        support_y = support_y[order]
        projections = projections[order]

        start_index = 0
        while start_index < len(projections):
            end_index = start_index
            while (
                end_index + 1 < len(projections)
                and projections[end_index + 1] - projections[end_index] <= max_line_gap
            ):
                end_index += 1

            if end_index > start_index:
                segment_length = projections[end_index] - projections[start_index]
                if segment_length >= min_line_len:
                    x1 = int(round(support_x[start_index]))
                    y1 = int(round(support_y[start_index]))
                    x2 = int(round(support_x[end_index]))
                    y2 = int(round(support_y[end_index]))
                    segments.append([[x1, y1, x2, y2]])

            start_index = end_index + 1

    if len(segments) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)
    return np.array(segments, dtype=np.int32)


def draw_lines(image: np.ndarray, lines: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 6) -> np.ndarray:
    canvas = np.asarray(image).copy()
    if canvas.ndim != 3 or canvas.shape[2] < 3:
        raise ValueError("draw_lines expects a 3-channel color canvas")

    if lines is None or len(lines) == 0:
        return canvas

    def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[Tuple[int, int]]:
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            double_error = 2 * error
            if double_error >= dy:
                error += dy
                x += step_x
            if double_error <= dx:
                error += dx
                y += step_y
        return points

    left_points = []
    right_points = []
    center_x = canvas.shape[1] / 2.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5:
            continue
        if slope < 0 and x1 < center_x and x2 < center_x:
            left_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and x1 > center_x and x2 > center_x:
            right_points.extend([(x1, y1), (x2, y2)])

    def fit_lane(points: list[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if len(points) < 2:
            return None
        xs = np.array([point[0] for point in points], dtype=np.float64)
        ys = np.array([point[1] for point in points], dtype=np.float64)
        slope, intercept = np.polyfit(xs, ys, 1)
        y_bottom = canvas.shape[0] - 1
        y_top = int(canvas.shape[0] * (1 - TRAP_HEIGHT))
        x_bottom = int(round((y_bottom - intercept) / slope))
        x_top = int(round((y_top - intercept) / slope))
        return x_bottom, y_bottom, x_top, y_top

    for lane_points in (left_points, right_points):
        fitted = fit_lane(lane_points)
        if fitted is None:
            continue
        x_bottom, y_bottom, x_top, y_top = fitted
        for x, y in bresenham_line(x_bottom, y_bottom, x_top, y_top):
            for row in range(max(0, y - thickness // 2), min(canvas.shape[0], y + thickness // 2 + 1)):
                for col in range(max(0, x - thickness // 2), min(canvas.shape[1], x + thickness // 2 + 1)):
                    canvas[row, col] = color

    return canvas


def draw_segments(image: np.ndarray, segments: np.ndarray, color: Tuple[int, int, int] = (255, 220, 60), thickness: int = 2) -> np.ndarray:
    canvas = _as_uint8_rgb(image).copy()
    if segments is None or len(segments) == 0:
        return canvas

    def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[Tuple[int, int]]:
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            double_error = 2 * error
            if double_error >= dy:
                error += dy
                x += step_x
            if double_error <= dx:
                error += dx
                y += step_y
        return points

    for segment in segments:
        x1, y1, x2, y2 = segment[0]
        for x, y in bresenham_line(x1, y1, x2, y2):
            for row in range(max(0, y - thickness), min(canvas.shape[0], y + thickness + 1)):
                for col in range(max(0, x - thickness), min(canvas.shape[1], x + thickness + 1)):
                    canvas[row, col] = color

    return canvas


def build_roi_vertices(height: int, width: int) -> np.ndarray:
    return np.array(
        [
            [
                ((width * (1 - TRAP_BOTTOM_WIDTH)) / 2, height),
                ((width * (1 - TRAP_TOP_WIDTH)) / 2, height * (1 - TRAP_HEIGHT)),
                (width - (width * (1 - TRAP_TOP_WIDTH)) / 2, height * (1 - TRAP_HEIGHT)),
                (width - (width * (1 - TRAP_BOTTOM_WIDTH)) / 2, height),
            ]
        ],
        dtype=np.int32,
    )


def annotate_image(image: np.ndarray, steps: Optional[Dict[str, bool]] = None) -> np.ndarray:
    rgb = _as_uint8_rgb(image)

    pipeline = {
        "color_filter": True,
        "blur": True,
        "canny": True,
        "roi": True,
        "hough": True,
        "overlay": True,
    }
    if steps:
        for key, value in steps.items():
            if key in pipeline:
                pipeline[key] = bool(value)

    filtered = filter_colors(rgb) if pipeline["color_filter"] else rgb
    gray = grayscale(filtered)
    blurred = gaussian_blur(gray, kernel_size=KERNEL_SIZE) if pipeline["blur"] else gray
    edges = canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD) if pipeline["canny"] else _as_uint8_gray(blurred)

    height, width = gray.shape
    vertices = build_roi_vertices(height, width)
    masked_edges = region_of_interest(edges, vertices) if pipeline["roi"] else edges
    segments = hough_lines(masked_edges) if pipeline["hough"] else np.empty((0, 1, 4), dtype=np.int32)

    lane_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    lane_canvas = draw_lines(lane_canvas, segments)

    if not pipeline["overlay"]:
        return lane_canvas

    alpha = 0.8
    beta = 1.0
    gamma = 0.0
    base = rgb.astype(np.float64)
    overlay = lane_canvas.astype(np.float64)
    annotated = np.clip(base * alpha + overlay * beta + gamma, 0, 255).astype(np.uint8)
    return annotated


def run_pipeline(image: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = _as_uint8_rgb(image)
    filtered = filter_colors(rgb)
    gray = grayscale(filtered)
    blurred = gaussian_blur(gray, kernel_size=KERNEL_SIZE)
    edges = canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD)
    vertices = build_roi_vertices(gray.shape[0], gray.shape[1])
    roi = region_of_interest(edges, vertices)
    segments = hough_lines(roi)
    lanes_only = draw_lines(np.zeros_like(rgb), segments)
    annotated = annotate_image(rgb)
    hough_overlay = draw_segments(rgb, segments)

    return {
        "input": rgb,
        "filtered": filtered,
        "gray": gray,
        "blur": blurred,
        "edges": edges,
        "roi": roi,
        "hough_overlay": hough_overlay,
        "lanes_only": lanes_only,
        "annotated": annotated,
        "segments": np.array([len(segments)], dtype=np.int32),
    }


def annotate_image_array(frame: np.ndarray, steps: Optional[Dict[str, bool]] = None) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype.kind == "f":
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return annotate_image(arr, steps=steps)


def _make_annotated_clip(video: Any, steps: Optional[Dict[str, bool]] = None) -> Any:
    frame_transform = getattr(video, "image_transform", None)
    frame_fn = lambda frame: annotate_image_array(frame, steps=steps)

    if frame_transform is not None:
        return frame_transform(frame_fn)

    frame_transform = getattr(video, "fl_image", None)
    if frame_transform is None:
        raise AttributeError("MoviePy does not support frame transforms in this version.")
    return frame_transform(frame_fn)


def annotate_video(
    input_file: str,
    output_file: str,
    *,
    steps: Optional[Dict[str, bool]] = None,
    show_progress: bool = False,
    use_gpu: bool = False,
    threads: Optional[int] = None,
    codec: Optional[str] = None,
    preset: str = "ultrafast",
    scale: float = 1.0,
    fps: Optional[int] = None,
) -> None:
    if VideoFileClip is None:
        raise ImportError("MoviePy is required. Install it with: pip install moviepy")

    if imageio_ffmpeg is not None:
        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
        except Exception as exc:
            raise ImportError("FFmpeg is required. Install it with: pip install imageio-ffmpeg") from exc

    video = VideoFileClip(input_file, audio=False)

    if scale != 1.0:
        if hasattr(video, "resized"):
            video = video.resized(scale)
        elif hasattr(video, "resize"):
            video = video.resize(scale)

    annotated_video = None
    effective_threads = threads if threads is not None else max(1, os.cpu_count() or 1)
    target_codec = codec or ("h264_nvenc" if use_gpu else "libx264")
    logger = "bar" if show_progress else None
    output_fps = fps if fps is not None else getattr(video, "fps", None)

    try:
        annotated_video = _make_annotated_clip(video, steps=steps)
        try:
            annotated_video.write_videofile(
                output_file,
                audio=False,
                codec=target_codec,
                preset=preset,
                threads=effective_threads,
                logger=logger,
                fps=output_fps,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
        except Exception:
            if use_gpu and target_codec == "h264_nvenc":
                annotated_video = _make_annotated_clip(video, steps=steps)
                annotated_video.write_videofile(
                    output_file,
                    audio=False,
                    codec="libx264",
                    preset=preset,
                    threads=effective_threads,
                    logger=logger,
                    fps=output_fps,
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )
            else:
                raise
    finally:
        video.close()
        if annotated_video is not None:
            annotated_video.close()


def _stage_output(results: Dict[str, np.ndarray], stage_key: str) -> np.ndarray:
    if stage_key == "filtered":
        return results["filtered"]
    if stage_key == "gray":
        return _as_uint8_gray(results["gray"])
    if stage_key == "blur":
        return _as_uint8_gray(results["blur"])
    if stage_key == "edges":
        return results["edges"]
    if stage_key == "roi":
        return results["roi"]
    if stage_key == "hough_overlay":
        return results["hough_overlay"]
    if stage_key == "lanes_only":
        return results["lanes_only"]
    return results["annotated"]


def _render_image_workflow() -> None:
    stage_labels = {
        "Full pipeline (lane overlay)": "annotated",
        "Color filter": "filtered",
        "Grayscale": "gray",
        "Gaussian blur": "blur",
        "Canny edges": "edges",
        "ROI mask": "roi",
        "Hough segments": "hough_overlay",
        "Lane overlay only": "lanes_only",
    }

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    stage_choice = st.selectbox("Stage output", list(stage_labels.keys()))

    if uploaded is None:
        st.info("Upload an image to start.")
        return

    image_array, image_bytes = load_image_from_upload(uploaded)
    image_hash = _hash_bytes(image_bytes)

    col_input, col_output = st.columns(2, gap="large")
    with col_input:
        st.markdown("<div class=\"panel\"><h3>Input</h3></div>", unsafe_allow_html=True)
        st.image(image_array, caption=uploaded.name, use_container_width=True)
        st.caption(f"Resolution: {image_array.shape[1]} x {image_array.shape[0]}")

    if st.button("Run image pipeline", use_container_width=True):
        with st.spinner("Processing image"):
            results = run_pipeline(image_array)
        st.session_state["image_results"] = results
        st.session_state["image_hash"] = image_hash
        st.session_state["image_name"] = Path(uploaded.name).stem
        st.session_state["image_saved_stage"] = None
        st.success("Image processed.")

    results = st.session_state.get("image_results")
    if results is None or st.session_state.get("image_hash") != image_hash:
        return

    stage_key = stage_labels[stage_choice]
    output = _stage_output(results, stage_key)

    saved_stage = st.session_state.get("image_saved_stage")
    if saved_stage != stage_key:
        output_name = f"{st.session_state.get('image_name', 'image')}_{stage_key}_{_timestamp()}.png"
        output_path = OUTPUT_DIR / output_name
        output_bytes = _save_image_bytes(output, output_path)
        st.session_state["image_saved_stage"] = stage_key
        st.session_state["image_saved_path"] = output_path
        st.session_state["image_saved_bytes"] = output_bytes

    with col_output:
        st.markdown("<div class=\"panel\"><h3>Output</h3></div>", unsafe_allow_html=True)
        st.image(output, use_container_width=True)
        segments_count = int(results["segments"][0])
        st.caption(f"Segments detected: {segments_count}")
        saved_path = st.session_state.get("image_saved_path")
        if saved_path:
            st.caption(f"Saved to outputs/{saved_path.name}")
            st.download_button(
                label="Download output",
                data=st.session_state.get("image_saved_bytes", b""),
                file_name=saved_path.name,
                mime="image/png",
            )


def _render_video_workflow() -> None:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    option_label = st.selectbox("Processing option", list(VIDEO_OPTIONS.keys()))

    with st.expander("Advanced options"):
        threads = st.number_input(
            "Threads",
            min_value=1,
            max_value=max(1, os.cpu_count() or 1),
            value=max(1, os.cpu_count() or 1),
            step=1,
        )
        use_gpu = st.checkbox("Use GPU if available", value=False)
        preset = st.selectbox("Encoder preset", ["ultrafast", "fast", "medium"], index=0)

    if uploaded is None:
        st.info("Upload a video to start.")
        return

    st.markdown("<div class=\"panel\"><h3>Input</h3></div>", unsafe_allow_html=True)
    st.video(uploaded)

    if st.button("Run video pipeline", use_container_width=True):
        if VideoFileClip is None:
            st.error("MoviePy is not available. Install it with: pip install moviepy")
            return
        with st.spinner("Processing video"):
            input_name = Path(uploaded.name).stem
            input_path = OUTPUT_DIR / f"input_{input_name}_{_timestamp()}.mp4"
            input_path.write_bytes(uploaded.getvalue())

            option = VIDEO_OPTIONS[option_label]
            option_slug = option_label.split("-")[0].strip().lower().replace(" ", "_")
            output_path = OUTPUT_DIR / f"lane_{input_name}_{option_slug}_{_timestamp()}.mp4"

            try:
                annotate_video(
                    str(input_path),
                    str(output_path),
                    scale=option["scale"],
                    fps=option["fps"],
                    threads=int(threads),
                    use_gpu=use_gpu,
                    preset=preset,
                    show_progress=False,
                )
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
                return

        st.session_state["video_output_path"] = output_path
        st.success("Video processed.")

    output_path = st.session_state.get("video_output_path")
    if output_path and Path(output_path).exists():
        st.markdown("<div class=\"panel\"><h3>Output</h3></div>", unsafe_allow_html=True)
        st.video(str(output_path))
        st.caption(f"Saved to outputs/{Path(output_path).name}")
        st.download_button(
            label="Download output",
            data=Path(output_path).read_bytes(),
            file_name=Path(output_path).name,
            mime="video/mp4",
        )


def main() -> None:
    st.set_page_config(page_title="Lane Detector Studio", layout="wide")
    _inject_styles()

    st.markdown(
        """
<div class="hero">
  <h1>Lane Detector Studio</h1>
  <p class="muted">Classic lane-line detection for images and videos, with stage-by-stage outputs.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Controls")
        input_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
        st.markdown("---")
        st.caption("Tip: For faster video tests, use Option 4 and short clips.")

    if input_type == "Image":
        _render_image_workflow()
    else:
        _render_video_workflow()


if __name__ == "__main__":
    main()
