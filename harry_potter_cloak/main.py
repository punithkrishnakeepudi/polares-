import cv2
import numpy as np
import time
import os
from datetime import datetime


# ------------------------- Utils -------------------------
def capture_background(cap, num_frames=30, flip=True):
    """Capture stable background using median of frames."""
    print("Capturing background... Step away.")
    frames = []
    for _ in range(num_frames):
        ret, bg = cap.read()
        if not ret:
            continue
        if flip:
            bg = np.flip(bg, axis=1)
        frames.append(bg)
        time.sleep(0.02)
    if not frames:
        return None
    return np.median(frames, axis=0).astype(np.uint8)


def make_color_ranges_from_avg(avg_hsv, dh=15, s_floor=50, v_floor=50):
    """Build HSV ranges around avg value with improved parameters."""
    h, s, v = int(avg_hsv[0]), int(avg_hsv[1]), int(avg_hsv[2])
    lower_h, upper_h = h - dh, h + dh
    lower_s, lower_v = max(s_floor, s - 70), max(v_floor, v - 70)
    upper_s, upper_v = min(255, s + 50), min(255, v + 50)

    if lower_h < 0:
        return [(np.array([0, lower_s, lower_v]), np.array([upper_h % 180, upper_s, upper_v])),
                (np.array([180 + lower_h, lower_s, lower_v]), np.array([179, upper_s, upper_v]))]
    elif upper_h > 179:
        return [(np.array([0, lower_s, lower_v]), np.array([upper_h % 180, upper_s, upper_v])),
                (np.array([lower_h, lower_s, lower_v]), np.array([179, upper_s, upper_v]))]
    else:
        return [(np.array([lower_h, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))]


def train_color_from_center(hsv, patch_half=20):
    """Train cloak color from center patch with larger sampling area."""
    h, w = hsv.shape[:2]
    cx, cy = w // 2, h // 2
    region = hsv[cy - patch_half:cy + patch_half, cx - patch_half:cx + patch_half]
    if region.size == 0:
        return None, None
    avg = np.mean(region, axis=(0, 1))
    return make_color_ranges_from_avg(avg), avg


def improve_mask_quality(mask):
    """Enhanced mask processing for better edge detection."""
    # Multiple kernel sizes for better morphological operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    # Remove noise with opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Fill holes with closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

    # Smooth edges
    mask = cv2.dilate(mask, kernel_medium, iterations=2)
    mask = cv2.erode(mask, kernel_small, iterations=1)

    # Advanced edge smoothing with bilateral filter on mask
    mask_float = mask.astype(np.float32) / 255.0
    mask_smooth = cv2.bilateralFilter(mask_float, 9, 50, 50)
    mask = (mask_smooth * 255).astype(np.uint8)

    # Final Gaussian blur for ultra-smooth edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


def create_collage(original_frame, cloak_frame, title="Invisible Cloak Effect"):
    """Create before/after collage."""
    h, w = original_frame.shape[:2]

    # Create collage canvas
    collage = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    collage[:] = (40, 40, 40)

    # Add frames
    collage[:, :w] = original_frame
    collage[:, w + 20:] = cloak_frame

    # Add labels
    cv2.putText(collage, "BEFORE", (w // 2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(collage, "AFTER", (w + 20 + w // 2 - 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add title at bottom
    cv2.putText(collage, title, (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return collage


def save_photo(original_frame, cloak_frame, photos_folder):
    """Save photo collage."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"invisible_cloak_photo_{timestamp}.jpg"
    filepath = os.path.join(photos_folder, filename)

    collage = create_collage(original_frame, cloak_frame, "Invisible Cloak Photo")
    cv2.imwrite(filepath, collage)
    print(f"Photo saved: {filepath}")
    return filepath


# ------------------------- Setup -------------------------
output_folder = "Invisible_Cloak_Output"
photos_folder = os.path.join(output_folder, "Photos")
videos_folder = os.path.join(output_folder, "Videos")

# Create organized folder structure
os.makedirs(output_folder, exist_ok=True)
os.makedirs(photos_folder, exist_ok=True)
os.makedirs(videos_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible.")

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
panel_width = 300

# Video recording variables
recording_video = False
video_frames_original = []
video_frames_cloak = []

time.sleep(2)
background = capture_background(cap)
bg_status_text = "Background captured" if background is not None else "Background not captured"
color_ranges, last_color_avg = None, None
detected_color_hint = "Cloth color not trained"
last_event, last_event_time = "", time.time()


# ------------------------- Button Callback -------------------------
def click_event(event, x, y, flags, param):
    global color_ranges, last_color_avg, detected_color_hint, last_event, last_event_time

    if event == cv2.EVENT_LBUTTONDOWN:
        original_frame = param

        # Color training (click on main area)
        if x < frame_width:  # Only if clicking on main frame area
            hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
            ranges, avg = train_color_from_center(hsv)
            if ranges is not None:
                color_ranges, last_color_avg = ranges, avg
                h, s, v = int(avg[0]), int(avg[1]), int(avg[2])
                detected_color_hint = f"Trained color H:{h} S:{s} V:{v}"
                last_event, last_event_time = "Cloth color trained", time.time()
                print(f"Trained color → H:{h}, S:{s}, V:{v}")


# ------------------------- Main Loop -------------------------
prev_time = time.time()

# Create fullscreen window
cv2.namedWindow("Invisible Cloak", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Invisible Cloak", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)
    original_frame = frame.copy()  # Keep original for saving
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Enhanced mask creation if color is trained
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    if color_ranges:
        for lower, upper in color_ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, temp_mask)

        # Apply improved mask processing
        mask = improve_mask_quality(mask)

    # Enhanced cloak effect with better blending
    if background is not None:
        cloak = cv2.bitwise_and(background, background, mask=mask)
        rest = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

        # Smooth blending at edges
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        final_output = (cloak * mask_3d + rest * (1 - mask_3d)).astype(np.uint8)
    else:
        final_output = frame

    # Store frames for video recording
    if recording_video:
        video_frames_original.append(original_frame.copy())
        video_frames_cloak.append(final_output.copy())

    # ---------------- Info Panel ----------------
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    # FPS
    curr_time = time.time()
    fps = 1.0 / max(1e-6, (curr_time - prev_time))
    prev_time = curr_time

    cv2.putText(panel, "Invisible Cloak", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, f"FPS: {int(fps)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    # Technical process in sentences
    cv2.putText(panel, "Process:", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(panel, "Background is captured first.", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
    cv2.putText(panel, "Cloth color is trained and masked.", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180),
                1)

    # Status
    y = 220
    cv2.putText(panel, "Status:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 150), 2);
    y += 30
    cv2.putText(panel, bg_status_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1);
    y += 25
    cv2.putText(panel, detected_color_hint, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1);
    y += 25

    if time.time() - last_event_time < 4 and last_event:
        cv2.putText(panel, f"Event: {last_event}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1);
        y += 25

    # Enhanced Controls
    y += 10
    cv2.putText(panel, "Controls:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 150), 2);
    y += 30
    cv2.putText(panel, "C - Capture cloth color", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1);
    y += 25
    cv2.putText(panel, "B - Recapture background", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1);
    y += 25
    cv2.putText(panel, "P - Take photo", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1);
    y += 25
    cv2.putText(panel, "V - Start/Stop video", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1);
    y += 25
    cv2.putText(panel, "Q - Quit", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1);
    y += 25

    # Recording indicator
    if recording_video:
        cv2.putText(panel, "● REC", (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Folders info
    y += 40
    cv2.putText(panel, "Output:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 150), 2);
    y += 30
    cv2.putText(panel, "Photos -> Photos/", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1);
    y += 20
    cv2.putText(panel, "Videos -> Videos/", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

    # Footer
    cv2.putText(panel, "Inspired by Harry Potter", (20, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    display = np.hstack((final_output, panel))
    cv2.imshow("Invisible Cloak", display)

    # Enable click events
    cv2.setMouseCallback("Invisible Cloak", click_event, original_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges, avg = train_color_from_center(hsv)
        if ranges is not None:
            color_ranges, last_color_avg = ranges, avg
            h, s, v = int(avg[0]), int(avg[1]), int(avg[2])
            detected_color_hint = f"Trained color H:{h} S:{s} V:{v}"
            last_event, last_event_time = "Cloth color captured (C)", time.time()
    elif key == ord('b'):
        background = capture_background(cap)
        if background is not None:
            bg_status_text = "Background captured"
            last_event, last_event_time = "Background recaptured (B)", time.time()
        else:
            bg_status_text = "Background not captured"
    elif key == ord('p'):
        # Take photo
        save_photo(original_frame, final_output, photos_folder)
        last_event, last_event_time = "Photo captured (P)", time.time()
    elif key == ord('v'):
        # Start/Stop video recording
        if not recording_video:
            recording_video = True
            video_frames_original = []
            video_frames_cloak = []
            last_event, last_event_time = "Video recording started (V)", time.time()
            print("Video recording started...")
        else:
            recording_video = False
            if video_frames_original and video_frames_cloak:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"invisible_cloak_video_{timestamp}.avi"
                filepath = os.path.join(videos_folder, filename)

                # Create video writer for collage
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                collage_height, collage_width = video_frames_original[0].shape[:2]
                collage_width = collage_width * 2 + 20
                out = cv2.VideoWriter(filepath, fourcc, 30, (collage_width, collage_height))

                # Write collage frames
                for orig, cloak in zip(video_frames_original, video_frames_cloak):
                    collage = create_collage(orig, cloak, "Invisible Cloak Video")
                    out.write(collage)

                out.release()
                print(f"Video saved: {filepath}")
                last_event, last_event_time = "Video saved (V)", time.time()

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Session ended. Check output folders:")
print(f"  Photos: {photos_folder}")
print(f"  Videos: {videos_folder}")
