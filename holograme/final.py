import os
import subprocess
from flask import Flask, request, send_from_directory, redirect, url_for, Response
from werkzeug.utils import secure_filename
from datetime import datetime

# -------- config --------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
PORT = 5001

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# -------- helpers --------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def ffprobe_size(path):
    """Return (width, height) of input video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w_str, h_str = out.split("x")
    return int(w_str), int(h_str)

def has_nvenc():
    """Detect h264_nvenc encoder availability."""
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
        return "h264_nvenc" in out
    except Exception:
        return False

def run_hologram_pyramid(input_path, output_path, panel=None, crf=18):
    """
    Build ghost pyramid cross layout:
      layout canvas = 3*panel x 3*panel
      positions:
         top -> (panel, 0) rotated 180
         left -> (0, panel) rotated +90
         right -> (2*panel, panel) rotated -90
         bottom -> (panel, 2*panel) no rotation
    If panel is None, auto-calc from input size (use shortest side // 3).
    """
    # probe input size
    iw, ih = ffprobe_size(input_path)
    short = min(iw, ih)

    # choose panel size
    if panel is None:
        # keep some headroom; panel fits 3x vertically into the short side
        panel = max(128, short // 3)

    canvas = panel * 3

    # safety clamps
    panel = max(64, panel)
    canvas = panel * 3

    # Choose encoder
    use_nvenc = False  # force disable NVENC
    vcodec = "libx264"
    v_preset = ["-preset", "veryfast"]

    if use_nvenc:
        v_preset = ["-preset", "fast"]  # compatible with FFmpeg 4.x
    else:
        v_preset = ["-preset", "veryfast"]

    # FFmpeg filter graph:
    # 1. scale input to panel x panel, split to 4
    # 2. rotate each appropriately and overlay on black canvas
    filter_complex = (
        f"[0:v]scale={panel}:{panel},split=4[v0][v1][v2][v3];"
        f"[v0]rotate=PI[vt];"        # top (180)
        f"[v1]rotate=PI/2[vl];"      # left (+90)
        f"[v2]rotate=-PI/2[vr];"     # right (-90)
        f"[v3]copy[vb];"             # bottom (0)
        f"color=size={canvas}x{canvas}:c=black[bg];"
        f"[bg][vt]overlay=x={panel}:y=0[bg1];"
        f"[bg1][vl]overlay=x=0:y={panel}[bg2];"
        f"[bg2][vr]overlay=x={panel*2}:y={panel}[bg3];"
        f"[bg3][vb]overlay=x={panel}:y={panel*2}[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",  # keep audio if present
        "-c:v", vcodec, *v_preset,
        "-pix_fmt", "yuv420p"
    ]

    if use_nvenc:
        # NVENC quick settings, decent quality; tune as needed
        cmd += ["-rc", "constqp", "-qp", "21"]
    else:
        # libx264 quality
        cmd += ["-crf", str(crf), "-tune", "fastdecode"]

    # encode audio to AAC (compatible)
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", output_path]

    # run ffmpeg
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # bubble ffmpeg stderr for debugging
        raise RuntimeError(f"FFmpeg failed (rc={proc.returncode}):\n{proc.stderr}")

# -------- routes --------

@app.route("/", methods=["GET"])
def index():
    html = """
    <!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Ghost Pyramid Hologram Converter</title>
  <style>
    body {
      font-family: 'Inter', Arial, sans-serif;
      background: linear-gradient(135deg, #111827, #1f2937);
      color: #f9fafb;
      min-height: 100vh;
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .card {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 40px;
      border-radius: 16px;
      box-shadow: 0 8px 40px rgba(0,0,0,0.5);
      text-align: center;
      max-width: 420px;
      width: 90%;
      backdrop-filter: blur(12px);
    }
    h2 {
      font-size: 1.6rem;
      margin-bottom: 16px;
    }
    input[type=file], input[type=number] {
      margin: 10px 0;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.2);
      background: rgba(255,255,255,0.1);
      color: #f9fafb;
      width: 100%;
      font-size: 0.95rem;
    }
    input::placeholder {
      color: #9ca3af;
    }
    button {
      margin-top: 16px;
      background: linear-gradient(90deg, #06b6d4, #3b82f6);
      color: white;
      font-weight: 600;
      padding: 12px 20px;
      border-radius: 10px;
      border: none;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.2s;
      width: 100%;
    }
    button:hover {
      transform: scale(1.03);
      opacity: 0.9;
    }
    .info {
      font-size: 0.85rem;
      color: #d1d5db;
      margin-top: 12px;
    }
    .warn {
      color: #fbbf24;
      font-size: 0.85rem;
      margin-top: 8px;
    }
    #msg {
      margin-top: 16px;
      font-size: 0.9rem;
    }
    .spinner {
      margin-top: 16px;
      display: none;
    }
    .spinner:after {
      content: "";
      display: block;
      width: 28px;
      height: 28px;
      margin: 0 auto;
      border-radius: 50%;
      border: 3px solid #3b82f6;
      border-color: #3b82f6 transparent #06b6d4 transparent;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
  <script>
    function onSubmit() {
      document.getElementById('msg').innerText = '⏳ Processing on server (GPU if available)...';
      document.querySelector('.spinner').style.display = 'block';
    }
  </script>
</head>
<body>
  <div class="card">
    <h2>Ghost Pyramid Hologram Converter</h2>
    <form action="/upload" method="post" enctype="multipart/form-data" onsubmit="onSubmit()">
      <input type="file" name="file" accept="video/*" required>
      <label>
        <input type="number" name="panel" placeholder="Panel size (px, optional)" min="64" step="1">
      </label>
      <button type="submit">✨ Convert Video</button>
    </form>
    <div id="msg" class="info"></div>
    <div class="spinner"></div>
    <p class="info">Layout: top (180°), left (+90°), right (-90°), bottom (0°). Audio preserved.</p>
  </div>
</body>
</html>
    """
    return Response(html, mimetype="text/html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400
    if not allowed_file(file.filename):
        return "Invalid file type", 400

    panel_val = request.form.get("panel", type=int)
    # Save uploaded file
    original_filename = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    saved_filename = f"{timestamp}_{original_filename}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
    file.save(upload_path)

    output_basename = f"holo_{os.path.splitext(original_filename)[0]}_{timestamp}.mp4"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_basename)

    try:
        run_hologram_pyramid(upload_path, output_path, panel=panel_val)
    except Exception as e:
        # return error output so you can debug ffmpeg issues
        return Response(f"<h3>Processing failed</h3><pre>{str(e)}</pre>", status=500, mimetype="text/html")

    return redirect(url_for("download_file", filename=output_basename))

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

# -------- main --------

if __name__ == "__main__":
    print("NVENC available:", has_nvenc())
    print(f"Serving on http://127.0.0.1:{PORT} (uploads -> {UPLOAD_FOLDER}, outputs -> {OUTPUT_FOLDER})")
    app.run(debug=True, port=PORT)