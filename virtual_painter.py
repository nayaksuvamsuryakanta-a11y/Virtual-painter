"""
Virtual Painter - draw in mid-air with your index finger (OpenCV + MediaPipe)
------------------------------------------------------------------------------
Controls (hand gestures):
    Index finger only up         -> DRAW
    Index + middle finger up     -> SELECT (touch a colour tile in the top bar;
                                            hold on "Clear" for 1 second to wipe)
    Anything else (fist, etc.)   -> pen up

Keys:
    + / -   bigger / smaller brush (or eraser, whichever is active)
    s       save drawing as PNG (on a white background)
    c       clear the canvas
    d       toggle debug overlay (shows the finger-extension numbers)
    q / Esc quit

Install:   pip install opencv-python mediapipe numpy
The first run downloads a ~8 MB hand model (hand_landmarker.task) next to this file.
Works with old AND new MediaPipe versions (the old `mp.solutions` API was removed
in 0.10.31, so this script uses the current Tasks API instead).
"""

import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720      # requested; the camera may choose differently
WINDOW_NAME = "Virtual Painter"

HEADER_HEIGHT = 100
DEFAULT_BRUSH = 8
DEFAULT_ERASER = 50
MIN_SIZE, MAX_SIZE = 2, 100

# Gesture tuning. Each number is (distance wrist->fingertip) / (distance wrist->middle joint).
# A straight finger measures ~1.3-1.4, a curled one ~0.6-0.8. Press 'd' to see live values.
INDEX_UP_RATIO = 1.15       # index counts as "up" above this
MIDDLE_UP_RATIO = 1.22      # middle counts as "up" above this (stricter -> fewer accidental selects)

MODE_CONFIRM_FRAMES = 2     # frames a new gesture must persist before the mode switches
ABSENT_RESET_FRAMES = 4     # hand missing this long -> forget the current mode
GAP_BRIDGE_FRAMES = 3       # a stroke survives this many bad frames without breaking
CLEAR_HOLD_SECONDS = 1.0
SAVE_DEBOUNCE_SECONDS = 1.0

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")

PALETTE = [                 # BGR colours, as OpenCV expects
    ("Blue", (255, 0, 0)),
    ("Green", (0, 255, 0)),
    ("Red", (0, 0, 255)),
    ("Yellow", (0, 255, 255)),
    ("Eraser", (90, 90, 90)),
    ("Clear", (40, 40, 40)),
]
PALETTE_COLORS = dict(PALETTE)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------
def build_palette_bounds(width, palette=PALETTE):
    bounds = []
    for index, (name, color) in enumerate(palette):
        bounds.append((name, color,
                       index * width // len(palette),
                       (index + 1) * width // len(palette)))
    return bounds


def palette_hit_test(point, bounds, header_height):
    x, y = point
    if y < 0 or y >= header_height:
        return None
    for name, _, x1, x2 in bounds:
        if x1 <= x < x2:
            return name
    return None


# ---------------------------------------------------------------------------
# Gesture recognition
# ---------------------------------------------------------------------------
def extension_ratio(pts, tip_id, pip_id):
    """How far past its middle joint the fingertip reaches, measured from the wrist.

    Uses x, y AND z in the same (pixel-like) units, so it still works when your
    finger points toward the camera. Measuring from the wrist (a big, stable
    distance) avoids the tiny-denominator problem of measuring from the knuckle.
    """
    wrist = pts[0]
    tip_d = np.linalg.norm(pts[tip_id] - wrist)
    pip_d = max(np.linalg.norm(pts[pip_id] - wrist), 1e-6)
    return float(tip_d / pip_d)


def classify_gesture(pts):
    """Return (gesture, index_ratio, middle_ratio); gesture is 'draw', 'select' or None."""
    index_ratio = extension_ratio(pts, 8, 6)
    middle_ratio = extension_ratio(pts, 12, 10)
    index_up = index_ratio > INDEX_UP_RATIO
    middle_up = middle_ratio > MIDDLE_UP_RATIO
    if index_up and not middle_up:
        gesture = "draw"
    elif index_up and middle_up:
        gesture = "select"
    else:
        gesture = None
    return gesture, index_ratio, middle_ratio


class ModeFilter:
    """Debounces the per-frame gesture so one noisy frame can't flip or lose the mode."""

    def __init__(self, confirm=MODE_CONFIRM_FRAMES, absent_reset=ABSENT_RESET_FRAMES):
        self.confirm = confirm
        self.absent_reset = absent_reset
        self.mode = None
        self.pending = None
        self.count = 0
        self.absent = 0

    def update(self, gesture, hand_present):
        if not hand_present:
            self.absent += 1
            if self.absent > self.absent_reset:
                self.mode, self.pending, self.count = None, None, 0
            return self.mode
        self.absent = 0
        if gesture is None:                 # unclear pose: neither confirms nor cancels
            return self.mode
        if gesture == self.mode:            # still the same mode: drop any pending switch
            self.pending, self.count = None, 0
            return self.mode
        if gesture == self.pending:
            self.count += 1
        else:
            self.pending, self.count = gesture, 1
        if self.count >= self.confirm:
            self.mode, self.pending, self.count = gesture, None, 0
        return self.mode


def smooth_point(previous, raw, min_alpha=0.35, max_alpha=0.85, speed_ref=60.0):
    """Adaptive smoothing: steady hand -> smooth line, fast motion -> almost no lag."""
    if previous is None:
        return (float(raw[0]), float(raw[1]))
    dx, dy = raw[0] - previous[0], raw[1] - previous[1]
    speed = (dx * dx + dy * dy) ** 0.5
    alpha = min_alpha + (max_alpha - min_alpha) * min(speed / speed_ref, 1.0)
    return (previous[0] + dx * alpha, previous[1] + dy * alpha)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def put_text(frame, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_header(frame, active_tool, bounds, header_height):
    for name, color, x1, x2 in bounds:
        cv2.rectangle(frame, (x1, 0), (x2 - 1, header_height - 1), color, -1)
        label_color = (0, 0, 0) if name == "Yellow" else (255, 255, 255)
        cv2.putText(frame, name, (x1 + 20, min(60, header_height - 5)),
                    FONT, 0.7, label_color, 2, cv2.LINE_AA)
        if name == active_tool:
            cv2.rectangle(frame, (x1, 0), (x2 - 1, header_height - 1), (255, 255, 255), 4)


def draw_skeleton(frame, pts):
    xy = [(int(p[0]), int(p[1])) for p in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], (200, 200, 200), 1, cv2.LINE_AA)
    for point in xy:
        cv2.circle(frame, point, 3, (60, 60, 60), cv2.FILLED)


# ---------------------------------------------------------------------------
# Hand tracking (MediaPipe Tasks API)
# ---------------------------------------------------------------------------
def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    print("Downloading hand-tracking model (~8 MB, first run only)...")
    part = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, part)
        part.replace(MODEL_PATH)
    except Exception as error:
        print(f"Could not download the model: {error}\n"
              f"Download it manually from:\n  {MODEL_URL}\n"
              f"and save it as:\n  {MODEL_PATH}")
        sys.exit(1)


class HandTracker:
    """detect(frame_bgr) -> (21, 3) float array of landmarks (x, y in pixels, z ~ pixels), or None."""

    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((time.monotonic() - self._t0) * 1000)
        if timestamp <= self._last_ts:          # VIDEO mode needs strictly increasing stamps
            timestamp = self._last_ts + 1
        self._last_ts = timestamp
        result = self._landmarker.detect_for_video(image, timestamp)
        if not result.hand_landmarks:
            return None
        return np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[0]],
                        dtype=np.float32)

    def close(self):
        self._landmarker.close()


# ---------------------------------------------------------------------------
# The painter: all per-frame logic lives here
# ---------------------------------------------------------------------------
class Painter:
    def __init__(self):
        self.canvas = None
        self.bounds = None
        self.tool = "Red"
        self.color = PALETTE_COLORS["Red"]
        self.sizes = {"brush": DEFAULT_BRUSH, "eraser": DEFAULT_ERASER}
        self.filter = ModeFilter()
        self.prev = None                    # last smoothed point of the current stroke
        self.gap = 0
        self.clear_start = None
        self.clear_done = False
        self.debug = False
        self.gesture = None
        self.idx_ratio = self.mid_ratio = 0.0
        self.fps = 0.0
        self._last_t = None
        self._last_save = -1e9
        self.toast, self.toast_until = "", 0.0

    @property
    def thickness(self):
        return self.sizes["eraser" if self.tool == "Eraser" else "brush"]

    def _ensure_canvas(self, w, h):
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), np.uint8)
        elif self.canvas.shape[:2] != (h, w):
            new = np.zeros((h, w, 3), np.uint8)
            ch, cw = min(h, self.canvas.shape[0]), min(w, self.canvas.shape[1])
            new[:ch, :cw] = self.canvas[:ch, :cw]
            self.canvas, self.prev = new, None
        if self.bounds is None or self.bounds[-1][3] != w:
            self.bounds = build_palette_bounds(w)

    def _apply_tool(self, name):
        if name is None or name == "Clear" or name == self.tool:
            return
        self.tool = name
        self.color = (0, 0, 0) if name == "Eraser" else PALETTE_COLORS[name]

    def _paint(self, tip):
        current = smooth_point(self.prev, tip)
        p1 = (int(round(current[0])), int(round(current[1])))
        p0 = p1 if self.prev is None else (int(round(self.prev[0])), int(round(self.prev[1])))
        t = self.thickness
        cv2.line(self.canvas, p0, p1, self.color, t)
        cv2.circle(self.canvas, p1, t // 2, self.color, cv2.FILLED)   # so a still finger leaves a dot
        self.prev = current

    def adjust_size(self, delta):
        key = "eraser" if self.tool == "Eraser" else "brush"
        self.sizes[key] = min(max(self.sizes[key] + delta, MIN_SIZE), MAX_SIZE)

    def save(self, now):
        if now - self._last_save < SAVE_DEBOUNCE_SECONDS:
            return
        self._last_save = now
        out = np.full_like(self.canvas, 255)
        mask = self.canvas.any(axis=2)
        out[mask] = self.canvas[mask]
        name = f"drawing_{int(time.time() * 1000)}.png"
        try:
            ok = cv2.imwrite(name, out)
        except Exception as error:
            print(f"Save error: {error}")
            ok = False
        self.toast = f"Saved {name}" if ok else "Save failed - check folder permissions"
        self.toast_until = now + 2.0
        print(self.toast)

    def handle_key(self, key, now):
        """Returns True when the program should quit."""
        if key in (ord("q"), 27):
            return True
        if key == ord("c") and self.canvas is not None:
            self.canvas.fill(0)
            self.prev = None
        elif key in (ord("+"), ord("=")):
            self.adjust_size(+4)
        elif key in (ord("-"), ord("_")):
            self.adjust_size(-4)
        elif key == ord("s") and self.canvas is not None:
            self.save(now)
        elif key == ord("d"):
            self.debug = not self.debug
        return False

    def step(self, frame, pts, now):
        """Process one mirrored BGR frame. `pts` is the landmark array or None."""
        h, w = frame.shape[:2]
        self._ensure_canvas(w, h)
        header_h = min(HEADER_HEIGHT, h)

        # --- 1. what is the hand doing? -------------------------------------
        gesture, tip = None, None
        self.idx_ratio = self.mid_ratio = 0.0
        if pts is not None:
            gesture, self.idx_ratio, self.mid_ratio = classify_gesture(pts)
            tip = (int(min(max(pts[8][0], 0), w - 1)), int(min(max(pts[8][1], 0), h - 1)))
        self.gesture = gesture
        mode = self.filter.update(gesture, pts is not None)
        # Act only when this frame agrees with the confirmed mode. While a switch is
        # pending, or the pose is unclear, the pen is simply up for that frame.
        action = mode if (gesture is not None and gesture == mode) else None

        # --- 2. act on it ---------------------------------------------------
        hovered = None
        if action == "select":
            self.prev, self.gap = None, 0
            hovered = palette_hit_test(tip, self.bounds, header_h)
            self._apply_tool(hovered)
        elif action == "draw":
            self.gap = 0
            if tip[1] < header_h:               # finger is over the toolbar: lift the pen
                self.prev = None
            else:
                self._paint(tip)
        else:
            self.gap += 1
            if self.gap > GAP_BRIDGE_FRAMES:    # short glitches are bridged, long ones end the stroke
                self.prev = None

        clear_progress = 0.0
        if hovered == "Clear":
            if self.clear_start is None:
                self.clear_start, self.clear_done = now, False
            clear_progress = min((now - self.clear_start) / CLEAR_HOLD_SECONDS, 1.0)
            if clear_progress >= 1.0 and not self.clear_done:
                self.canvas.fill(0)
                self.prev, self.clear_done = None, True
        else:
            self.clear_start, self.clear_done = None, False

        # --- 3. render ------------------------------------------------------
        painted = self.canvas.any(axis=2)
        frame[painted] = self.canvas[painted]
        if pts is not None:
            draw_skeleton(frame, pts)
        draw_header(frame, self.tool, self.bounds, header_h)

        if tip is not None:
            t = self.thickness
            if action == "draw" and tip[1] >= header_h:
                pos = self.prev if self.prev is not None else tip
                pos = (int(pos[0]), int(pos[1]))
                if self.tool == "Eraser":
                    cv2.circle(frame, pos, t // 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pos, max(t // 2, 3), self.color, cv2.FILLED)
                    cv2.circle(frame, pos, max(t // 2, 3), (255, 255, 255), 1, cv2.LINE_AA)
            elif action == "select":
                ring = self.color if self.color != (0, 0, 0) else (180, 180, 180)
                cv2.circle(frame, tip, 14, (255, 255, 255), cv2.FILLED)
                cv2.circle(frame, tip, 14, ring, 3)
            else:
                cv2.circle(frame, tip, 10, (0, 200, 255), 2, cv2.LINE_AA)   # pen up

        if 0.0 < clear_progress < 1.0:
            x1, x2 = self.bounds[-1][2], self.bounds[-1][3]
            cv2.rectangle(frame, (x1, max(0, header_h - 8)),
                          (x1 + int((x2 - x1) * clear_progress), header_h - 1), (255, 255, 255), -1)

        if self._last_t is not None:
            instant = 1.0 / max(now - self._last_t, 1e-6)
            self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant
        self._last_t = now

        label = {"draw": "DRAW", "select": "SELECT"}.get(action) or \
                ("PEN UP" if pts is not None else "NO HAND")
        put_text(frame, f"{label} | Tool: {self.tool} | Size: {self.thickness} | FPS: {int(self.fps)}",
                 (10, max(20, h - 15)), 0.65, (0, 255, 0), 2)
        put_text(frame, "Index: draw | Index+Middle: select | +/-: size | s: save | c: clear | d: debug | q: quit",
                 (10, min(header_h + 28, h - 5)), 0.55)
        if self.debug:
            put_text(frame, f"gesture={self.gesture}  index={self.idx_ratio:.2f} (up>{INDEX_UP_RATIO})  "
                            f"middle={self.mid_ratio:.2f} (up>{MIDDLE_UP_RATIO})",
                     (10, max(20, h - 45)), 0.6, (0, 255, 255))
        if now < self.toast_until:
            put_text(frame, self.toast, (max(10, w - 430), max(20, h - 15)), 0.6, (0, 255, 255), 2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def open_camera():
    cap = None
    if sys.platform.startswith("win"):          # DirectShow opens much faster than the default on Windows
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def main():
    ensure_model()
    cap = open_camera()
    if not cap.isOpened():
        print("Error: could not access the webcam (is another app using it? try CAMERA_INDEX = 1).")
        cap.release()
        return

    tracker = HandTracker()
    painter = Painter()
    failed_reads = frames = 0
    check_window = None                         # decided once, after the window has appeared

    try:
        while True:
            ok, frame = cap.read()
            if not ok:                          # tolerate the odd dropped frame at start-up
                failed_reads += 1
                if failed_reads > 30:
                    print("Error: the webcam stopped delivering frames.")
                    break
                time.sleep(0.01)
                continue
            failed_reads = 0
            frames += 1

            frame = cv2.flip(frame, 1)          # mirror so it behaves like a mirror
            pts = tracker.detect(frame)
            now = time.monotonic()
            painter.step(frame, pts, now)

            cv2.imshow(WINDOW_NAME, frame)
            if painter.handle_key(cv2.waitKey(1) & 0xFF, now):
                break

            # Quit when the window's X button is clicked (skipped on backends that can't report it).
            if frames >= 10:
                try:
                    visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    visible = 0
                if check_window is None:
                    check_window = visible >= 1
                elif check_window and visible < 1:
                    break
    except KeyboardInterrupt:
        print("Interrupted - closing.")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()