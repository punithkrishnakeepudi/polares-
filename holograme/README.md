# Ghost Pyramid Hologram Converter

This is a Flask-based web application that converts standard video files into a four-panel layout suitable for ghost pyramid hologram displays. The layout arranges the video in a cross pattern (top, left, right, bottom) with appropriate rotations for each panel.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed.
- **FFmpeg**: Install FFmpeg with ffprobe. Download from [FFmpeg's official site](https://ffmpeg.org/download.html) or use a package manager:
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS (Homebrew): `brew install ffmpeg`
  - Windows: Follow instructions on FFmpeg's site or use a package manager like Chocolatey.
- **Optional**: NVIDIA GPU with NVENC support for hardware-accelerated encoding (detected automatically).

## Installation

1. Clone or download this repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure FFmpeg is installed and accessible in your system’s PATH.

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```
   The server will start at `http://127.0.0.1:5001`.
2. Open a web browser and navigate to `http://127.0.0.1:5001`.
3. Upload a video file (supported formats: MP4, MOV, AVI, MKV, WebM).
4. Optionally specify a panel size (in pixels) or leave blank for auto-calculation.
5. Click "Convert Video" to process the video and download the result.

## How It Works

- The application uses FFmpeg to process videos into a 3x3 grid layout:
  - Top panel: Rotated 180°
  - Left panel: Rotated +90°
  - Right panel: Rotated -90°
  - Bottom panel: No rotation
- Audio is preserved if present in the input video.
- The output is an MP4 file using the libx264 encoder (or h264_nvenc if available).
- Uploaded files are stored in the `uploads` folder, and processed files are stored in the `outputs` folder.

## Notes

- Ensure sufficient disk space for input and output files.
- For large videos, processing may take time depending on hardware.
- If you encounter FFmpeg errors, check the error message in the browser for debugging.
- NVENC is disabled by default but can be enabled by modifying the `use_nvenc` variable in the code.

## License

MIT License
