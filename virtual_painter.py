"""
Virtual Painter - draw in mid-air with your index finger (OpenCV + MediaPipe)
Includes professional-grade geometric shape snapping with live preview.

Controls (hand gestures):
  Index finger only up         -> DRAW
  Index + middle finger up     -> SELECT (touch a colour tile; hold "Clear" 1 s to wipe)
  Anything else (fist, etc.)   -> pen up

Shape snapping (press 'n' to toggle):
  Draw a rough line / circle / rectangle / triangle, then hold your fingertip
  still for ~0.5 s. A yellow preview shows the detected perfect shape while a
  green ring fills around your fingertip; when the ring completes, the freehand
  ink is replaced by the mathematically perfect primitive.

Keys:
  + / -   bigger / smaller brush
  s       save drawing as PNG (white background)
  c       clear the canvas (undoable)
  u / z   undo the last stroke or clear
  n       toggle shape snapping
  d       toggle debug overlay
  q / Esc quit

Install:   pip install opencv-python mediapipe numpy
The first run downloads a ~8 MB hand model (hand_landmarker.task) next to this file.
Works with old AND new MediaPipe versions (uses the current Tasks API).
"""

import math
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
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
WINDOW_NAME = "Virtual Painter"
HEADER_HEIGHT = 100
DEFAULT_BRUSH = 8
DEFAULT_ERASER = 50
MIN_SIZE, MAX_SIZE = 2, 100

INDEX_UP_RATIO = 1.15
MIDDLE_UP_RATIO = 1.22
MODE_CONFIRM_FRAMES = 2
ABSENT_RESET_FRAMES = 4
GAP_BRIDGE_FRAMES = 3
CLEAR_HOLD_SECONDS = 1.0
SAVE_DEBOUNCE_SECONDS = 1.0
UNDO_LIMIT = 10

# Shape snapping tuning ------------------------------------------------------
SHAPE_SNAP_DEFAULT = True
SHAPE_MIN_POINTS = 24
SHAPE_MIN_CHORD = 80.0
SHAPE_MIN_AREA = 800.0
SHAPE_HOLD_SECONDS = 0.45
SHAPE_STILL_PX = 6.0
SHAPE_RESAMPLE_N = 64
SHAPE_CIRCLE_MIN = 0.82
SHAPE_ANGLE_TOL_DEG = 14.0
SHAPE_TRIANGLE_MIN_DEG = 25.0
SHAPE_LINE_MIN = 0.96
SHAPE_SQUARE_RATIO = 0.85

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
# FIX: Corrected __file__ dunder
MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")
SAVE_DIR = Path(__file__).resolve().parent

PALETTE = [
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
    wrist = pts[0]
    tip_d = np.linalg.norm(pts[tip_id] - wrist)
    pip_d = max(np.linalg.norm(pts[pip_id] - wrist), 1e-6)
    return float(tip_d / pip_d)

def classify_gesture(pts):
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
        if gesture is None:
            return self.mode
        if gesture == self.mode:
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
    if previous is None:
        return (float(raw[0]), float(raw[1]))
    dx, dy = raw[0] - previous[0], raw[1] - previous[1]
    speed = (dx * dx + dy * dy) ** 0.5
    alpha = min_alpha + (max_alpha - min_alpha) * min(speed / speed_ref, 1.0)
    return (previous[0] + dx * alpha, previous[1] + dy * alpha)

# ---------------------------------------------------------------------------
# Shape snapping: geometric analysis
# ---------------------------------------------------------------------------
def _denoise(points, window=3):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < window + 2:
        return pts
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(pts[:, 0], kernel, mode="same"),
                    np.convolve(pts[:, 1], kernel, mode="same")], axis=1)
    out[0], out[-1] = pts[0], pts[-1]
    return out

def resample_points(points, n):
    pts = np.asarray(points, dtype=np.float64)
    keep = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
    pts = pts[keep]
    if len(pts) < 2:
        return [tuple(pts[0])] * n
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    if total <= 1e-9:
        return [tuple(pts[0])] * n
    targets = np.linspace(0.0, total, n)
    return list(zip(np.interp(targets, cum, pts[:, 0]),
                    np.interp(targets, cum, pts[:, 1])))

def _corner_angles_deg(poly):
    m = len(poly)
    angles = []
    for i in range(m):
        a = np.asarray(poly[i - 1], dtype=np.float64)
        b = np.asarray(poly[i], dtype=np.float64)
        c = np.asarray(poly[(i + 1) % m], dtype=np.float64)
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cosang = min(1.0, max(-1.0, float(np.dot(v1, v2) / (n1 * n2))))
        angles.append(math.degrees(math.acos(cosang)))
    return angles

def classify_shape(stroke):
    if len(stroke) < SHAPE_MIN_POINTS:
        return None

    pts = _denoise(stroke)
    p0, p1 = np.asarray(pts[0]), np.asarray(pts[-1])
    chord = float(np.linalg.norm(p1 - p0))
    if chord < SHAPE_MIN_CHORD:
        return None

    res = resample_points(pts, SHAPE_RESAMPLE_N)
    cnt = np.array(res, dtype=np.float32).reshape(-1, 1, 2)
    closed_peri = cv2.arcLength(cnt, True)
    
    # FIX: Guard against division by zero
    if closed_peri <= 1e-6:
        return None
        
    arc = closed_peri - chord
    if arc <= 1e-6:
        return None

    if chord / arc >= SHAPE_LINE_MIN:
        return "line", (tuple(p0.astype(int)), tuple(p1.astype(int))), "Line"

    area = cv2.contourArea(cnt)
    if area < SHAPE_MIN_AREA:
        return None

    circularity = 4.0 * math.pi * area / (closed_peri * closed_peri)
    if circularity >= SHAPE_CIRCLE_MIN and len(res) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (_cx, _cy), (ax1, ax2), _ang = ellipse
        if ax1 > 20 and ax2 > 20:
            ratio = min(ax1, ax2) / max(ax1, ax2)
            label = "Circle" if ratio >= SHAPE_SQUARE_RATIO else "Ellipse"
            return "circle", ellipse, label

    approx = cv2.approxPolyDP(cnt, 0.02 * closed_peri, True)
    poly = approx.reshape(-1, 2)
    if len(poly) in (3, 4) and cv2.isContourConvex(
            np.asarray(poly, np.float32).reshape(-1, 1, 2)):
        angles = _corner_angles_deg(poly)
        if angles is not None:
            if len(poly) == 3 and min(angles) >= SHAPE_TRIANGLE_MIN_DEG:
                return "triangle", poly.astype(int), "Triangle"
            if len(poly) == 4 and all(abs(a - 90.0) <= SHAPE_ANGLE_TOL_DEG for a in angles):
                box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)
                d1 = float(np.linalg.norm(box[0] - box[1]))
                d2 = float(np.linalg.norm(box[1] - box[2]))
                ratio = min(d1, d2) / max(max(d1, d2), 1e-6)
                label = "Square" if ratio >= SHAPE_SQUARE_RATIO else "Rectangle"
                return "rect", box, label
    return None

def draw_shape(canvas, shape, color, thickness):
    kind, payload, _label = shape
    if kind == "line":
        cv2.line(canvas, payload[0], payload[1], color, thickness, cv2.LINE_AA)
    elif kind == "circle":
        cv2.ellipse(canvas, payload, color, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [payload], True, color, thickness, cv2.LINE_AA)

def draw_shape_preview(frame, shape):
    kind, payload, _label = shape
    color = (0, 255, 255)
    if kind == "line":
        cv2.line(frame, payload[0], payload[1], color, 2, cv2.LINE_AA)
    elif kind == "circle":
        cv2.ellipse(frame, payload, color, 2, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [payload], True, color, 2, cv2.LINE_AA)

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
        if timestamp <= self._last_ts:
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
# The whiteboard: all per-frame logic lives here
# ---------------------------------------------------------------------------
class Painter:
    # FIX: Corrected __init__ dunder
    def __init__(self):
        self.canvas = None
        self.bounds = None
        self.tool = "Red"
        self.color = PALETTE_COLORS["Red"]
        self.sizes = {"brush": DEFAULT_BRUSH, "eraser": DEFAULT_ERASER}
        self.filter = ModeFilter()
        self.prev = None
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
        self.history = []
        self.post_snap_lock = False  # FIX 1: Post-snap lock initialization
        
        # shape snapping state
        self.snap = SHAPE_SNAP_DEFAULT
        self.stroke = []
        self.stroke_snapshot = None
        self.hold_start = None
        self.last_raw_tip = None
        self.snapped = False
        self.shape_preview = None

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

    def _push_history(self):
        # FIX 2: Deduplicate history to prevent spam on gesture flicker
        if self.history and np.array_equal(self.history[-1], self.canvas):
            return
        self.history.append(self.canvas.copy())
        if len(self.history) > UNDO_LIMIT:
            self.history.pop(0)

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
        cv2.circle(self.canvas, p1, t // 2, self.color, cv2.FILLED)
        self.prev = current

    def _end_stroke(self):
        self.prev = None
        self.stroke = []
        self.stroke_snapshot = None
        self.hold_start = None
        self.snapped = False
        self.shape_preview = None

    def _maybe_snap(self, tip, now):
        if not self.snap or self.tool == "Eraser" or self.snapped:
            self.shape_preview = None
            return
            
        moved = math.hypot(tip[0] - self.last_raw_tip[0], tip[1] - self.last_raw_tip[1]) if self.last_raw_tip is not None else 0
        self.last_raw_tip = tip
        
        if moved > SHAPE_STILL_PX:
            self.hold_start = None
            self.shape_preview = None
            return
            
        if self.hold_start is None:
            self.hold_start = now
            
        self.shape_preview = classify_shape(self.stroke)
        if now - self.hold_start < SHAPE_HOLD_SECONDS:
            return
            
        if self.shape_preview is None or self.stroke_snapshot is None:
            return
            
        self.snapped = True
        self.post_snap_lock = True  # FIX 1: Engage lock to prevent immediate stray stroke
        self.canvas[:] = self.stroke_snapshot
        draw_shape(self.canvas, self.shape_preview, self.color, self.thickness)
        self.prev = None
        self.stroke_snapshot = self.canvas.copy()
        self.toast = f"Snapped: {self.shape_preview[2]}"
        self.toast_until = now + 2.0
        print(self.toast)
        self.shape_preview = None

    def undo(self):
        if not self.history:
            return False
        self.canvas[:] = self.history.pop()
        self._end_stroke()
        return True

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
        save_path = SAVE_DIR / name
        try:
            ok = cv2.imwrite(str(save_path), out)
        except Exception as error:
            print(f"Save error: {error}")
            ok = False
        self.toast = f"Saved {name}" if ok else "Save failed - check folder permissions"
        self.toast_until = now + 2.0
        print(self.toast)

    def handle_key(self, key, now):
        if key in (ord("q"), 27):
            return True
        if key == ord("c") and self.canvas is not None:
            self._push_history()
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
        elif key == ord("n"):
            self.snap = not self.snap
            self.toast = f"Shape snap: {'ON' if self.snap else 'OFF'}"
            self.toast_until = now + 2.0
        elif key in (ord("u"), ord("z")):
            self.toast = "Undo" if self.undo() else "Nothing to undo"
            self.toast_until = now + 2.0
        return False

    def step(self, frame, pts, now):
        h, w = frame.shape[:2]
        self._ensure_canvas(w, h)
        header_h = min(HEADER_HEIGHT, h)

        gesture, tip = None, None
        self.idx_ratio = self.mid_ratio = 0.0
        if pts is not None:
            gesture, self.idx_ratio, self.mid_ratio = classify_gesture(pts)
            tip = (int(min(max(pts[8][0], 0), w - 1)), int(min(max(pts[8][1], 0), h - 1)))
        self.gesture = gesture
        mode = self.filter.update(gesture, pts is not None)
        action = mode if (gesture is not None and gesture == mode) else None

        hovered = None
        if action == "select":
            self.gap = 0
            self._end_stroke()
            hovered = palette_hit_test(tip, self.bounds, header_h)
            self._apply_tool(hovered)
        elif action == "draw":
            self.gap = 0
            if tip[1] < header_h:
                self._end_stroke()
            else:
                if self.prev is None:
                    # FIX 1: Check lock before starting a new stroke
                    if self.post_snap_lock:
                        self.last_raw_tip = tip
                    else:
                        self._push_history()
                        self.stroke_snapshot = self.history[-1]
                        self.stroke = [tip]
                        self.hold_start = None
                        self.snapped = False
                        self.last_raw_tip = tip
                        self.shape_preview = None
                else:
                    self.stroke.append(tip)
                self._paint(tip)
                self._maybe_snap(tip, now)
        else:
            self.gap += 1
            self.post_snap_lock = False  # FIX 1: Release lock when pen is lifted
            if self.gap > GAP_BRIDGE_FRAMES:
                self._end_stroke()

        clear_progress = 0.0
        if hovered == "Clear":
            if self.clear_start is None:
                self.clear_start, self.clear_done = now, False
            clear_progress = min((now - self.clear_start) / CLEAR_HOLD_SECONDS, 1.0)
            if clear_progress >= 1.0 and not self.clear_done:
                self._push_history()
                self.canvas.fill(0)
                self.prev, self.clear_done = None, True
        else:
            self.clear_start, self.clear_done = None, False

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
                cv2.circle(frame, tip, 10, (0, 200, 255), 2, cv2.LINE_AA)

        if self.shape_preview is not None and self.hold_start is not None:
            draw_shape_preview(frame, self.shape_preview)
            put_text(frame, self.shape_preview[2], (tip[0] + 20, max(20, tip[1] - 14)),
                     0.6, (0, 255, 255), 2)
            progress = min((now - self.hold_start) / SHAPE_HOLD_SECONDS, 1.0)
            cv2.ellipse(frame, (tip[0], tip[1]), (26, 26), 0, -90,
                        int(-90 + 360 * progress), (0, 255, 0), 2, cv2.LINE_AA)

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
        put_text(frame, "Index: draw | Index+Middle: select | +/-: size | s: save | c: clear | "
                        "u: undo | n: snap | d: debug | q: quit",
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
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        # FIX 4: Release failed handle before fallback
        if cap is not None and not cap.isOpened():
            cap.release()
            cap = None
            
    if cap is None:
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
    check_window = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > 30:
                    print("Error: the webcam stopped delivering frames.")
                    break
                time.sleep(0.01)
                continue
            failed_reads = 0
            frames += 1
            frame = cv2.flip(frame, 1)
            pts = tracker.detect(frame)
            now = time.monotonic()
            painter.step(frame, pts, now)
            cv2.imshow(WINDOW_NAME, frame)
            if painter.handle_key(cv2.waitKey(1) & 0xFF, now):
                break
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

# FIX: Corrected __name__ dunder
if __name__ == "__main__":
    main()