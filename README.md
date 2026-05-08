# Lane Detector

A classical computer-vision pipeline for detecting road lane lines in images and videos. The project includes a step-by-step notebook and a clean Streamlit UI for interactive use.

## Highlights

- Transparent, classical lane-detection pipeline.
- Notebook walkthrough with visuals at every stage.
- Streamlit app for image and video processing.
- Outputs are saved automatically for download and reuse.

## Project Structure

- `P1.ipynb` - full pipeline with explanations and visuals.
- `streamlit_app.py` - interactive UI for images and videos.
- `test_images/` - sample images.
- `*.mp4` and `*.jpg` - sample videos and result assets.

## Pipeline Stages

1. Color filtering for white/yellow lanes
2. Grayscale conversion
3. Gaussian blur
4. Canny edge detection
5. Region of interest mask (trapezoid)
6. Hough line segments
7. Lane line fitting and overlay

## Requirements

- Python 3.11+
- numpy
- matplotlib
- streamlit
- pillow
- moviepy
- imageio-ffmpeg
- Optional: opencv-python

## Quick Start

### Notebook

1. Open `P1.ipynb`.
2. Run cells in order.

### Streamlit app

Install dependencies:

```bash
pip install streamlit pillow moviepy imageio-ffmpeg numpy matplotlib
```

Run the app:

```bash
streamlit run streamlit_app.py
```

## Streamlit Features

### Image Mode

- Upload an image.
- Choose the output stage:
	- Full pipeline (lane overlay)
	- Color filter
	- Grayscale
	- Gaussian blur
	- Canny edges
	- ROI mask
	- Hough segments
	- Lane overlay only
- Output is saved to `outputs/` and available for download.

### Video Mode

- Upload a video.
- Choose a processing option:
	- Option 1 - Full quality (slowest)
	- Option 2 - Faster: reduced FPS only
	- Option 3 - Faster: half resolution
	- Option 4 - Fastest: half res + low FPS
- Output is saved to `outputs/` and previewed in the app.

## Output Files

All generated artifacts are stored in `outputs/` with timestamps. Example names:

- `image_annotated_20240508_153000.png`
- `lane_challenge_option_1_20240508_154000.mp4`

## Troubleshooting

- FFmpeg errors: install imageio-ffmpeg (or system FFmpeg) and restart the app.
- MoviePy missing: `pip install moviepy`.
- Slow processing: use Option 4 and short clips for quick testing.

## Repository Goal

This project is educational: it demonstrates a full, transparent lane-detection pipeline without a black-box model.
