import threading
from flask import Flask, jsonify

import math
import queue
import sys
import time

import numpy as np
import pygame
import sounddevice as sd
import cv2
import mediapipe as mp

from collections import deque

# ---- Flask API + shared game_data ----
app = Flask(__name__)

game_data = {
    "angle": "--.-°",
    "volume": 0.0,
}
_game_data_lock = threading.Lock()

@app.route("/data")
def get_data():
    with _game_data_lock:
        return jsonify({"angle": game_data["angle"], "volume": game_data["volume"]})

def _run_flask():
    # no reloader so it doesn't spawn a second process
    app.run(host="127.0.0.1", port=8080, threaded=True, use_reloader=False)


db_window = deque()
window_s = 1


# ===================== Audio =====================
def pick_sample_rate():
    try:
        info = sd.query_devices(kind='input')
        return int(info.get('default_samplerate', 44100) or 44100)
    except Exception:
        return 44100

SAMPLE_RATE = pick_sample_rate()
BLOCKSIZE = 1024
CHANNELS = 1
A_REF = 1.0
MIN_DB = -60.0

audio_q = queue.Queue(maxsize=8)

def audio_callback(indata, frames, time_info, status):
    if status:
        # You could print(status) for debugging; ignored to keep console clean
        pass
    rms = float(np.sqrt(np.mean(np.square(indata[:, 0], dtype=np.float64))))
    try:
        audio_q.put_nowait(rms)
    except queue.Full:
        pass

def dbfs_from_rms(rms):
    if rms <= 0.0:
        return MIN_DB
    return max(20.0 * math.log10(rms / A_REF), MIN_DB)

def norm_from_db(db):
    return (db - MIN_DB) / (-MIN_DB)

# ===================== Pygame UI =====================
WIDTH, HEIGHT = 640, 260
BAR_MARGIN = 24

def draw_meter(screen, norm_level, db_text, peak_norm, threshold_db, font):
    screen.fill((15, 15, 20))

    # Track
    track_rect = pygame.Rect(BAR_MARGIN, HEIGHT // 2 - 20, WIDTH - 2 * BAR_MARGIN, 40)
    pygame.draw.rect(screen, (50, 55, 70), track_rect, border_radius=10)

    # Fill
    fill_w = int(track_rect.width * max(0.0, min(1.0, norm_level)))
    fill_rect = pygame.Rect(track_rect.left, track_rect.top, fill_w, track_rect.height)
    pygame.draw.rect(screen, (90, 170, 255), fill_rect, border_radius=10)

    # Peak hold
    peak_x = int(track_rect.left + track_rect.width * max(0.0, min(1.0, peak_norm)))
    pygame.draw.line(screen, (255, 210, 110), (peak_x, track_rect.top - 6), (peak_x, track_rect.bottom + 6), width=2)

    # Threshold line
    thr_norm = norm_from_db(max(threshold_db, MIN_DB))
    thr_x = int(track_rect.left + track_rect.width * thr_norm)
    pygame.draw.line(screen, (255, 120, 120), (thr_x, track_rect.top - 12), (thr_x, track_rect.bottom + 12), width=2)

    # Ticks
    for d in [-60, -48, -36, -24, -12, -6, 0]:
        x = int(track_rect.left + track_rect.width * norm_from_db(max(d, MIN_DB)))
        pygame.draw.line(screen, (100, 105, 125), (x, track_rect.top - 10), (x, track_rect.top - 2), 1)
        label = font.render(f"{d}", True, (170, 175, 190))
        screen.blit(label, (x - label.get_width() // 2, track_rect.top - 28))

    # Labels
    title = font.render("Mic Loudness (dBFS, 0 = max digital level)", True, (200, 205, 220))
    screen.blit(title, (BAR_MARGIN, 16))
    value = font.render(db_text, True, (235, 240, 255))
    screen.blit(value, (BAR_MARGIN, HEIGHT - 70))
    thr_lbl = font.render(f"Threshold: {threshold_db:.1f} dBFS  (↑/↓ adjust, R reset peak, D devices, Q quit)", True, (220, 200, 200))
    screen.blit(thr_lbl, (BAR_MARGIN, HEIGHT - 40))

# ===================== MediaPipe Pose =====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def to_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def visible(landmark, thresh=0.5):
    return (getattr(landmark, "visibility", 1.0)) >= thresh

# ===================== Combined App =====================
def main():
    # ---- Pygame init ----
    pygame.init()
    pygame.display.set_caption("Mic Loudness Meter + Jump Trigger")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    # ---- Audio stream ----
    ema = 0.0
    alpha = 0.35
    peak_norm = 0.0
    peak_decay_per_sec = 0.35

    threshold_db = -23.0
    hysteresis_db = 2.0
    cooldown_s = 0.30
    armed = True
    last_fire = 0.0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        channels=CHANNELS,
        dtype='float32',
        callback=audio_callback,
    )
    try:
        stream.start()
    except Exception as e:
        pygame.quit()
        print("Could not start microphone stream. Check mic permissions.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    # ---- Camera init ----
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        # try default index
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        stream.stop(); stream.close()
        pygame.quit()
        raise SystemExit("Error: Could not open webcam. Check camera permissions and index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Pose
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    last_time = time.time()
    running = True
    last_angle_signed = None  # store most recent valid angle for printout
    
    # Start Flask HTTP server in the background
    threading.Thread(target=_run_flask, daemon=True).start()

    while running:
        # ---- Handle Pygame events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    try:
                        devs = sd.query_devices()
                        print("\n=== Audio Devices ===")
                        for i, d in enumerate(devs):
                            print(f"[{i}] {d['name']}  (in:{d['max_input_channels']}, out:{d['max_output_channels']})")
                        print("=====================\n")
                    except Exception as e:
                        print("Device query failed:", e, file=sys.stderr)
                elif event.key == pygame.K_r:
                    peak_norm = 0.0
                elif event.key == pygame.K_UP:
                    threshold_db = min(0.0, threshold_db + 1.0)
                elif event.key == pygame.K_DOWN:
                    threshold_db = max(MIN_DB, threshold_db - 1.0)
                elif event.key == pygame.K_q:
                    running = False

        # ---- Get latest audio RMS ----
        rms = None
        try:
            while True:
                rms = audio_q.get_nowait()
        except queue.Empty:
            pass

        if rms is not None:
            ema = alpha * rms + (1 - alpha) * ema

        db = dbfs_from_rms(ema)
        norm = norm_from_db(db)

        # Peak update
        now = time.time()
        dt = now - last_time
        last_time = now
        peak_norm = max(peak_norm * math.exp(-peak_decay_per_sec * dt), norm)

        # ---- Camera frame & pose ----
        ok, frame_bgr = cap.read()
        if not ok:
            print("Error: Failed to grab frame.")
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Defaults for drawing/text
        angle_text = "Angle: --.-°"
        line_color = (255, 255, 0)     # cyan
        dot_color  = (0, 165, 255)     # orange
        connect_color = (128, 0, 255)  # violet
        line_thk   = 3
        dot_radius = 6

        if results.pose_landmarks is not None:
            # Skeleton
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            lm = results.pose_landmarks.landmark
            L = mp_pose.PoseLandmark

            ls, rs = lm[L.LEFT_SHOULDER], lm[L.RIGHT_SHOULDER]
            lh, rh = lm[L.LEFT_HIP], lm[L.RIGHT_HIP]

            mid_shoulder = None
            mid_hip = None

            if visible(ls) and visible(rs):
                ls_px = to_px(ls, w, h)
                rs_px = to_px(rs, w, h)
                cv2.line(frame_bgr, ls_px, rs_px, line_color, line_thk)
                mid_shoulder = ((ls_px[0] + rs_px[0]) // 2, (ls_px[1] + rs_px[1]) // 2)
                cv2.circle(frame_bgr, mid_shoulder, dot_radius, dot_color, -1)

            if visible(lh) and visible(rh):
                lh_px = to_px(lh, w, h)
                rh_px = to_px(rh, w, h)
                cv2.line(frame_bgr, lh_px, rh_px, line_color, line_thk)
                mid_hip = ((lh_px[0] + rh_px[0]) // 2, (lh_px[1] + rh_px[1]) // 2)
                cv2.circle(frame_bgr, mid_hip, dot_radius, dot_color, -1)

            if (mid_shoulder is not None) and (mid_hip is not None):
                # Connect
                cv2.line(frame_bgr, mid_shoulder, mid_hip, connect_color, line_thk)

                # Angle vs vertical (signed: right lean positive)
                dx = mid_shoulder[0] - mid_hip[0]
                dy = mid_shoulder[1] - mid_hip[1]
                angle_rad = math.atan2(abs(dx), max(1e-6, abs(dy)))
                angle_deg = math.degrees(angle_rad)
                signed_angle = angle_deg * (1 if dx > 0 else (-1 if dx < 0 else 0))
                last_angle_signed = signed_angle
                angle_text = f"Angle vs vertical: {signed_angle:+.1f}°"

                # Vertical reference at hip
                ref_len = max(40, int(0.08 * h))
                p1 = (mid_hip[0], mid_hip[1] - ref_len)
                p2 = (mid_hip[0], mid_hip[1] + ref_len)
                cv2.line(frame_bgr, p1, p2, (200, 200, 200), 2)

        # ---- Trigger: print when crossing threshold ----
        
        
        now = time.time()
        db_window.append((now, db))
        
        while db_window and (now - db_window[0][0]) > window_s:
            db_window.popleft()

        recent_peak = max(v for _, v in db_window) if db_window else MIN_DB

        with _game_data_lock:
            game_data["angle"] = f"{last_angle_signed:+.1f}°" if last_angle_signed is not None else "--.-°"
            game_data["volume"] = round(recent_peak, 1)  # serve recent max
        
        
        if armed and db >= threshold_db and (now - last_fire) >= cooldown_s:
            ang_str = "--.-"
            if last_angle_signed is not None:
                ang_str = f"{last_angle_signed:+.1f}°"
            print(f"jump!!!!!!!  angle={ang_str}  level={db:.1f} dBFS")
            
            with _game_data_lock:
                game_data["angle"] = ang_str
                game_data["volume"] = round(db, 1)
            
            last_fire = now
            armed = False
        if not armed and db <= (threshold_db - hysteresis_db):
            armed = True
        

        # ---- Draw Pygame meter ----
        draw_meter(screen, norm, f"{db:5.1f} dBFS", peak_norm, threshold_db, font)
        pygame.display.flip()

        # ---- Overlay on camera and show ----
        cv2.rectangle(frame_bgr, (10, 10), (460, 60), (0, 0, 0), -1)
        cv2.putText(frame_bgr, angle_text, (18, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Body Tracker (midpoints + angle vs vertical)", frame_bgr)
        # Allow 'q' to quit from the OpenCV window
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            running = False

        clock.tick(60)

    # ---- Cleanup ----
    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    stream.stop()
    stream.close()
    pygame.quit()

if __name__ == "__main__":
    main()

