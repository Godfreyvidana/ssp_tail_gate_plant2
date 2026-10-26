# -*- coding: utf-8 -*-
"""
Panel Vision Inspection System
Vidana Consulting Pvt Ltd

Hardware:  NVIDIA GPU / CPU only
Model:     models/SSP_TAILGATE_SEQ_PLANT2-V1.pt   (PyTorch YOLO)  → primary
Fallback:  models/SSP_TAILGATE_SEQ_PLANT2-V1.pt   (PyTorch YOLO)  → CPU if GPU unavailable

Image capture per panel:
  SEQ1  →  full frame  +  left / middle / right / bottom crops
  SEQ2  →  full frame only
  SEQ3  →  full frame only
  SEQ4  →  full frame  +  serial-number ROI (rotated upright for OCR)

PDF generated automatically once all 4 sequences complete.
"""

import os
import torch  # ADDED: import torch before paddle to fix DLL load error on Windows
os.environ['ULTRALYTICS_OFFLINE'] = 'True'
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

import cv2
def speak(text, lang='en'):
    """
    Non-blocking TTS via PowerShell System.Speech.
    Used for: missed-sequence alerts and any other short messages.
    """
    import subprocess
    print(f'[TTS] {text}')
    script = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{text}')"
    try:
        # Run completely detached, no window, no blocking
        subprocess.Popen(['powershell', '-Command', script], creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"[TTS Error] {e}")


def _announce_landscape(seq_num: int):
    """
    Speak landscape-placement instruction for SEQ2 or SEQ3.
    Plays English then Hindi.  Repeats once after 5 s if panel not yet
    captured (so operator doesn't miss it in a noisy factory).
    Non-blocking — runs entirely in a background thread.

    English : "Sequence 2. Please place the panel in landscape position"
    Hindi   : "Sequence do. Panel ko landscape position mein rakhein"
    """
    en_text = (f"Sequence {seq_num}. Please place the panel in landscape position")

    def _is_captured():
        if seq_num == 2:
            return getattr(state, 'seq2_auto_captured', False)
        if seq_num == 3:
            return getattr(state, 'seq3_auto_captured', False)
        return True

    def _play_once():
        import subprocess
        script = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{en_text}')"
        try:
            subprocess.run(['powershell', '-Command', script], creationflags=subprocess.CREATE_NO_WINDOW)
        except:
            pass

    def _run():
        if _is_captured(): return
        _play_once()
        # Repeat once after 6 seconds if still not captured
        time.sleep(6)
        if not _is_captured():
            _play_once()
    threading.Thread(target=_run, daemon=True).start()

import numpy as np
import subprocess
import threading
import time
import select
import gc
import _queue
import shutil

def _resolve_ffmpeg_bin():
    """Resolve the ffmpeg executable on Windows using PATH or common install folders."""
    env_path = os.environ.get('FFMPEG_BIN') or os.environ.get('FFMPEG_EXE')
    print(f'[RTSP] env FFMPEG_BIN={env_path!r}')
    if env_path:
        print(f'[RTSP] env path exists: {os.path.exists(env_path)} file: {os.path.isfile(env_path)}')
        if os.path.isfile(env_path):
            return env_path

    path = shutil.which('ffmpeg')
    print(f'[RTSP] shutil.which(ffmpeg)={path!r}')
    if path and os.path.isfile(path):
        return path

    if os.name == 'nt':
        try:
            where = subprocess.run(
                ['where', 'ffmpeg'], capture_output=True, text=True, timeout=4)
            print(f'[RTSP] where ffmpeg rc={where.returncode} out={where.stdout!r}')
            if where.returncode == 0:
                for line in where.stdout.splitlines():
                    candidate = line.strip()
                    print(f'[RTSP] where candidate={candidate!r} exists={os.path.isfile(candidate)}')
                    if candidate and os.path.isfile(candidate):
                        return candidate
        except Exception as e:
            print(f'[RTSP] where ffmpeg failed: {e}')

        candidates = [
            os.path.expandvars(r'%ProgramFiles%\ffmpeg\bin\ffmpeg.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\ffmpeg\bin\ffmpeg.exe'),
            r'C:\ffmpeg\bin\ffmpeg.exe',
        ]
        for candidate in candidates:
            print(f'[RTSP] candidate path={candidate!r} exists={os.path.isfile(candidate)}')
            if candidate and os.path.isfile(candidate):
                return candidate
    return None

# Locate ffmpeg binary once at startup so all subsystems can reuse it.
FFMPEG_BIN = _resolve_ffmpeg_bin()
if FFMPEG_BIN:
    print(f'[RTSP] ffmpeg resolved to: {FFMPEG_BIN}')
else:
    print('[RTSP] ffmpeg not found in PATH — ffmpeg-based RTSP capture disabled')

import re
import platform
from datetime import datetime
from collections import deque
from panel_logger import get_logger
log = get_logger()

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from pdf_generator import generate_pdf_report
# import pytesseract  # Removed: Migrated to EasyOCR for memory efficiency on Pi

# Camera-2 OCR module (standalone — no Camera-1 dependency)
try:
    from camera2_ocr import Camera2OCR
    _CAM2_OCR_AVAILABLE = True
except ImportError:
    _CAM2_OCR_AVAILABLE = False
    print('[WARN] camera2_ocr.py not found — Camera-2 OCR disabled')

# [REMOVED] Tesseract path configuration — Migrated to EasyOCR
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='.')
CORS(app)

# ── Remove Flask's built-in size limit completely.
# We handle large files by streaming directly to disk (see upload_video).
app.config['MAX_CONTENT_LENGTH'] = None

# Return proper JSON on 413 just in case any proxy sends one
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# CONFIGURATION & PERFORMANCE
# ─────────────────────────────────────────────────────────────
MAX_PROC_WIDTH      = 640     # YOLO inference width (GPU)
FRAME_SKIP          = 2       # Process every 3rd frame
JPEG_QUALITY        = 50      # [FREEZE FIX] 65→50: faster encode, less CPU per frame
OCR_THROTTLE_FRAMES = 20      # OCR every 20 frames only
STREAM_FPS_CAP      = 15      # [FREEZE FIX] 20→15: real-time feel, 33% less encoder work
STREAM_UI_WIDTH      = 960    # [FREEZE FIX] 1280→960: 44% fewer pixels per frame = major CPU saving

# ── Wiping completion — time-based ───────────────────────────
# How many seconds of continuous wiping = sequence complete
SEQ_WIPE_SECONDS    = 10.0   # Progress bar duration in seconds (from config)
SEQ_HAND_GONE_SECS  = 1.0    # Seconds hand must be absent before SEQ3 can complete

# ── NEW: Capture delay (seconds hand must be absent before snapshot) ──
CAPTURE_DELAY_SECS = 2.0   # Loaded from config: capture_delay_secs

# ── NEW: Progress bar total duration (seconds of wiping = 100%) ──────────
PROGRESS_BAR_DURATION_SECS = 10.0   # Loaded from config: progress_bar_duration_secs

# ── Sequence Timing Config ───────────────────────────────────
SEQ1_NORMAL_FRAMES  = 2      # Frames to wait for sequence 1 capture with serial
SEQ1_FALLBACK_FRAMES= 30     # Frames to wait for sequence 1 capture without serial

# True PyTorch/Ultralytics class order from training
# Actual model class names: hand, seq1, seq2, seq3, Serial
CLASS_NAMES = {
    0: 'hand',
    1: 'seq1',    # internal alias: panel_seq1
    2: 'seq2',    # internal alias: panel_seq2
    3: 'seq3',    # internal alias: panel_seq3
    4: 'Serial',  # internal alias: serial_number
}

# Reverse lookup: internal name → model class index
CLASS_INDEX = {
    'hand':          0,
    'panel_seq1':    1,
    'panel_seq2':    2,
    'panel_seq3':    3,
    'serial_number': 4,
    # Also accept raw model names directly
    'seq1':   1,
    'seq2':   2,
    'seq3':   3,
    'serial': 4,
}

# panel class name → inspection sequence number (1-based)
CLASS_MAP = {
    "panel_seq1": 1,
    "panel_seq2": 2,
    "panel_seq3": 3,
}

# Human-readable short labels for SEQ panels
SEQ_DISPLAY_NAMES = {
    'panel_seq1': 'SEQ1',
    'panel_seq2': 'SEQ2',
    'panel_seq3': 'SEQ3',
}

DISPLAY_MAP = {
    "panel_seq1":           "SEQ1",
    "panel_seq2":           "SEQ2",
    "panel_seq3":           "SEQ3",
    "hand":                 "HAND",
    "serial_number":        "SERIAL",
    "serial_number_region": "SERIAL",
}

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(SCRIPT_DIR, "models")
PT_MODEL_PATH   = os.path.join(MODELS_DIR, "SSP_TAILGATE_SEQ_PLANT2-V1.pt")
# Store panel images under the application directory, not the Pi path.
BASE_STORAGE    = os.path.join(SCRIPT_DIR, "panel_data")
TEMP_DIR       = os.path.join(SCRIPT_DIR, "temp_frames")

# ─────────────────────────────────────────────────────────────
# GRACE PERIOD TIMERS (after prompting operator)
# ─────────────────────────────────────────────────────────────
_SEQ2_GRACE_PERIOD = 10.0   # INCREASED to 10s — operator has plenty of time   # INCREASED: seconds after prompt before capturing (operator positioning time)
_SEQ3_GRACE_PERIOD = 10.0   # INCREASED to 10s — operator has plenty of time   # INCREASED: seconds after prompt before capturing (operator positioning time)
_SEQ2_COUNTDOWN_DURATION = 5.0  # Countdown timer before capture (1-5 seconds)
_SEQ3_COUNTDOWN_DURATION = 5.0  # Countdown timer before capture (1-5 seconds)

for d in (MODELS_DIR, BASE_STORAGE, os.path.join(BASE_STORAGE, '_uploads'), TEMP_DIR):
    os.makedirs(d, exist_ok=True)



# ─────────────────────────────────────────────────────────────
# CPU INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────

def normalize_class_name(raw_name: str) -> str:
    """Normalize model class names to internal names.
    
    Model outputs:  hand, seq1, seq2, seq3, Serial
    Internal names: hand, panel_seq1, panel_seq2, panel_seq3, serial_number
    """
    raw = str(raw_name).lower().strip()

    # ── Exact matches first (fastest path, covers actual model output) ──
    if raw == 'seq1':   return 'panel_seq1'
    if raw == 'seq2':   return 'panel_seq2'
    if raw == 'seq3':   return 'panel_seq3'
    if raw == 'serial': return 'serial_number'
    if raw == 'hand':   return 'hand'

    # ── Numeric index fallback (when model outputs class indices as strings) ──
    if raw == '0': return 'hand'
    if raw == '1': return 'panel_seq1'
    if raw == '2': return 'panel_seq2'
    if raw == '3': return 'panel_seq3'
    if raw == '4': return 'serial_number'

    # ── Substring fallback (covers variants like 'panel_seq1', 'panel1', etc.) ──
    if 'seq1' in raw or 'panel1' in raw or 'panel 1' in raw: return 'panel_seq1'
    if 'seq2' in raw or 'panel2' in raw or 'panel 2' in raw: return 'panel_seq2'
    if 'seq3' in raw or 'panel3' in raw or 'panel 3' in raw: return 'panel_seq3'
    if 'serial' in raw: return 'serial_number'
    if 'hand'   in raw: return 'hand'

    return raw_name  # unknown — pass through unchanged

class CPUInference:
    """YOLO inference — auto-selects CUDA GPU or CPU fallback."""
    def __init__(self, pt_path: str, force_cpu: bool = False):
        import torch
        from ultralytics import YOLO

        self._device = 'cpu'
        if not force_cpu and self._cuda_usable():
            self._device = 'cuda:0'

        self.model = YOLO(pt_path)
        self.model_names = getattr(self.model, 'names', {})

        try:
            self.model.predict(
                source=__import__('numpy').zeros((640, 640, 3), dtype='uint8'),
                device=self._device, verbose=False)
        except Exception as e:
            if self._device != 'cpu':
                print(f"  [WARN] CUDA warmup failed: {e}. Falling back to CPU.")
                self._device = 'cpu'
                self.model.predict(
                    source=__import__('numpy').zeros((640, 640, 3), dtype='uint8'),
                    device='cpu', verbose=False)
            else:
                raise

        mode = f'CUDA GPU ({self._device})' if self._device != 'cpu' else 'CPU'
        print(f"  ✅ YOLO ready — {pt_path}  [{mode}]")

    @staticmethod
    def _cuda_usable():
        import torch
        if not torch.cuda.is_available():
            return False
        try:
            torch.zeros(1, device='cuda:0')
            return True
        except Exception as e:
            print(f"  [WARN] CUDA available but unusable: {e}")
            return False

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.25):
        results = self.model.predict(
            source=frame,
            device=self._device,
            conf=conf_threshold,
            verbose=False)
        
        # Raw debug (Every frame for first 100 frames, then every 30)
        f_idx = getattr(state, 'frame_count', 0)
        if f_idx < 100 or f_idx % 30 == 0:
            if results and len(results[0].boxes) > 0:
                dets = [(self.model_names.get(int(b.cls[0]), f"class_{int(b.cls[0])}"), float(b.conf[0]))
                        for b in results[0].boxes]
                print(f"  [AI DEBUG] Frame {f_idx} | Detections: {dets}")
            elif f_idx % 30 == 0:
                print(f"  [AI DEBUG] Frame {f_idx} | No boxes found")

        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                raw_name = self.model_names.get(class_id, CLASS_NAMES.get(class_id, f"class_{class_id}"))
                name = normalize_class_name(raw_name)
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class_id': class_id,
                    'name': name,
                    'polygon': None
                })
        
        if f_idx % 30 == 0:
            print(f"[AI] Frame {f_idx} | Detections: {len(detections)}")
        return detections


# ─────────────────────────────────────────────────────────────
# ENGINE LOADER  — YOLO GPU / CPU only
# ─────────────────────────────────────────────────────────────

def load_inference_engine():
    """
    Smart priority — picks the FASTEST engine for the current hardware:

      Z440 / NVIDIA GPU  →  CUDA available           → GPU YOLO  (SSP_TAILGATE_SEQ_PLANT2-V1.pt)
      No CUDA              → CPU YOLO  (SSP_TAILGATE_SEQ_PLANT2-V1.pt)
    """
    import torch
    has_cuda = torch.cuda.is_available()
    device_name = 'N/A'
    if has_cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception as e:
            device_name = f'unavailable ({e})'

    print("\n[INFO] Loading inference engine ...")
    print(f"[DEBUG] Active CLASS_NAMES: {CLASS_NAMES}")
    print(f"[INFO] CUDA available: {has_cuda}  ({device_name})")

    if os.path.exists(PT_MODEL_PATH):
        try:
            engine = CPUInference(PT_MODEL_PATH)
            mode = "GPU" if engine._device != 'cpu' else "CPU"
            print(f"[OK] Running on {mode} — YOLO SSP_TAILGATE_SEQ_PLANT2-V1.pt\n")
            return engine, mode.lower()
        except Exception as e:
            print(f"  [ERR] YOLO load failed: {e}")
            if has_cuda:
                print("  [INFO] Retrying model load on CPU fallback...")
                try:
                    engine = CPUInference(PT_MODEL_PATH, force_cpu=True)
                    print("[OK] Running on CPU fallback — YOLO SSP_TAILGATE_SEQ_PLANT2-V1.pt\n")
                    return engine, "cpu"
                except Exception as e2:
                    print(f"  [ERR] CPU fallback failed: {e2}")
    else:
        print(f"  [ERR] {PT_MODEL_PATH} not found")

    print("[ERR] No inference engine available — hand detection disabled\n")
    return None, "none"


# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────

class PanelDetectionState:
    def __init__(self):
        self.camera_active        = False
        self.current_sequence = 0
        self.completed = {1: False, 2: False, 3: False}
        self.wipe_t0 = None
        self.progress = 0
        self.ocr_done = False
        self.serial = None
        self.last_clean = None
        self.orig = None
        from collections import deque
        self.seq_buf = deque(maxlen=5)
        self.wipe_buf = deque(maxlen=3)
        self.panel_id = 0
        self.status_msg = "IDLE"
        self.current_frame        = None
        self.panel_contour        = None
        self.panel_rect           = None
        self.panel_mask           = None
        self.cleaned_mask         = None
        self.current_sequence     = 0
        # status: 'active' | 'wiping' | 'completed' | 'pending' | 'missed'
        self.sequence_status      = {1: 'pending', 2: 'pending', 3: 'pending'}
        self.failure_reason       = None
        self.hand_bbox            = None
        self.prev_gray            = None
        self.motion_history       = deque(maxlen=10)
        self.panel_history        = deque(maxlen=5)
        self.last_panel_position  = None
        self.cleaning_percentage  = 0.0
        self.frame_count          = 0
        self.serial_number        = "UNKNOWN"
        self.sequence_captured    = {1: False, 2: False, 3: False}
        self.panel_lost_count     = 0       # Smoothing for overlays
        self.wiping_active        = False   # True when hand+motion detected
        self.current_panel_name   = None    # e.g., 'panel_seq1'
        self.ocr_frame_count      = 0
        self.serial_det           = None    # Storage for current model detection
        self.consecutive_wiping_frames = 0  # For smooth transitions
        self.seq3_absence_frames       = 0  # Counter for panel removal
        self.seq3_seen_frames          = 0  # Counter for panel presence
        self.seq1_wiping_frames   = 0
        self.seq2_wiping_frames   = 0
        self.seq3_wiping_frames   = 0
        self.no_wiping_frames     = 0       # For new panel detection
        self.panel_absent_frames  = 0       # Frames where no panel is detected
        self.ocr_candidates       = []      # Collection of OCR results for voting
        self.last_clean_frame        = None    # Best frame without hand for PDF
        self.last_clean_panel_rect   = None    # panel_rect at last clean frame
        self.last_clean_contour      = None    # contour at last clean frame
        self.last_clean_serial_det   = None    # serial det at last clean frame
        self.detected_seq_history = deque(maxlen=20) # Stronger debounce
        self.seq_capture_data     = {1: None, 2: None, 3: None}
        self.orig_frame           = None
        self.seq1_clean_images    = []
        self.max_ocr_images       = 5

        # ── Landscape alert (voice + screen colour for SEQ2/SEQ3) ────
        # Values: "" | "seq2" | "seq3" | "captured"
        self.landscape_alert        = ""
        self.landscape_alert_ts     = 0.0   # time alert was set
        self._seq2_cap_announced    = False  # True after "captured" voice played
        self._seq3_cap_announced    = False
        self._seq2_prompted         = False
        self._seq3_prompted         = False

        # ── Lead time tracking ────────────────────────────────────
        self.panel_start_time     = None          # when panel first detected
        self.seq_start_time       = {1: None, 2: None, 3: None}   # when each seq started
        self.seq_end_time         = {1: None, 2: None, 3: None}   # when each seq completed
        self.panel_end_time       = None          # when all seqs done
        self.current_sequence_panel_folder = None          # Locked folder for this panel
        self.panel_id             = None          # Unique ID for this panel session

        # ── Wipe time tracking (time-based completion) ────────────
        self.wipe_start_time      = None   # when continuous wiping started
        self.total_wipe_seconds   = 0.0    # accumulated wipe seconds this seq
        self.hand_gone_since      = None   # when hand last disappeared

        # ── SEQ3 completion tracking ──────────────────────────────
        # Track consecutive wiping/no-wiping frames for SEQ3
        self.seq3_wiping_frames   = 0      # Consecutive frames with wiping
        self.seq3_no_wipe_frames  = 0      # Consecutive frames without wiping after 5 wipes
        self.seq3_completion_stage = 0     # 0=waiting, 1=wiping, 2=checking_no_wipe

        # ── Auto-capture on sequence detection ──────────────────
        self.last_detected_seq    = 0      # Track which seq was last detected
        self.seq1_auto_captured   = False  # Has SEQ1 been auto-captured?
        self.seq2_auto_captured   = False  # Has SEQ2 been auto-captured?
        self.seq3_auto_captured   = False  # Has SEQ3 been auto-captured?
        # First unannotated frame the moment panel_seq1 appears
        self.seq1_first_clean_frame = None
        self.seq1_snapshot_data     = None   # Stores {'frame', 'rect', 'contour'} for SEQ1

        # ── NEW: Per-zone hand-free delay timer (must be absent CAPTURE_DELAY_SECS before snap) ──
        self.seq_hand_free_since    = {1: None, 2: None, 3: None}
        self.seq_delay_triggered    = {1: False, 2: False, 3: False}

        # ── NEW: Per-sequence phase tracking ────────────────────
        # phase: 'idle' | 'waiting_class' | 'hand_clear' | 'captured' | 'wiping' | 'done'
        self.seq_phase = {1: 'idle', 2: 'idle', 3: 'idle'}

        # ── NEW: Post-capture wipe seconds (drives progress bar 0→100%) ──
        self.post_capture_wipe_secs = {1: 0.0, 2: 0.0, 3: 0.0}
        self.post_capture_last_wipe = {1: None, 2: None, 3: None}

        # ── Best landscape frame trackers (initialize at startup) ────────────
        # These store the sharpest HORIZONTAL frame seen for each sequence.
        self.seq2_best_frame      = None
        self.seq2_best_sharp      = 0.0
        self.seq2_best_meta       = {}
        self.seq2_landscape_count = 0      # consecutive landscape frames seen
        self._seq2_countdown_time = None    # When prompt issued (for countdown display)
        self._seq3_countdown_time = None    # When prompt issued (for countdown display)
        self.seq2_miss_count      = 0      # consecutive non-landscape frames
        self.seq2_prev_center     = None   # (cx, cy) for stability check

        self.seq3_best_frame      = None
        self.seq3_best_sharp      = 0.0
        self.seq3_best_meta       = {}
        self.seq3_landscape_count = 0
        self.seq3_miss_count      = 0
        self.seq3_prev_center     = None   # (cx, cy) for stability check
        self.seq3_absence_frames  = 0
        self.seq3_seen_frames     = 0

        # ── SEQ1 stability counter (direct access in process_frame) ──────────
        self.seq1_detection_count   = 0    # consecutive frames with panel+serial
        self.seq2_stable_for_seq1   = 0    # consecutive frames SEQ2 active (gates SEQ1 completion)

        # ── Attributes set in reset_for_new_panel but missing from __init__ ──
        # All are accessed directly (not via getattr) in process_frame /
        # update_seq / update_wiping, so they MUST exist from construction.
        self.seq3_wiping_started    = False  # True once SEQ3 wiping begins
        self.was_absent_long        = False  # True if panel absent >30 frames
        self.ocr_triggered_by_tilt  = False  # True once portrait-tilt OCR fired
        self.seq2_stable_count      = 0      # stable-frame counter for SEQ2
        self.seq3_stable_count      = 0      # stable-frame counter for SEQ3
        self.seq2_in_sequence_frames = 0     # total frames spent in SEQ2

        # ── Wipe % captured at the moment of sequence transition (before reset) ──
        # Used by the SEQ1/SEQ2/SEQ3 completion gates to verify adequate wipe happened.
        self.seq1_final_wipe_pct    = 0      # SEQ1 progress% saved before SEQ2 reset
        self.seq2_final_wipe_pct    = 0      # SEQ2 progress% saved before SEQ3 reset
        self.seq2_final_wiping_frames = 0     # SEQ2 wiping frame count saved before reset
        self.seq2_detection_confidence = 0   # confidence of SEQ2 when first detected
        self.seq3_final_wipe_pct    = 0      # SEQ3 progress% saved before completion
        # FIX: panel_bbox_xyxy must be initialized — referenced by capture() via _meta()
        self.panel_bbox_xyxy        = None   # (x1, y1, x2, y2) of current panel bbox

        # ── Panel reset counter — increments every new panel ──────
        # Frontend watches this to know exactly when to reset bars
        self.panel_reset_id       = 0
        self.stream_generation    = 0
        self.local_camera_id      = 0
        self.current_sequence_panel_name = None
        self.last_wiping_time     = None
        self.model_sees           = 'NONE'

        # ── Daily panel counter ────────────────────────────────
        self.today_date           = datetime.now().strftime("%Y-%m-%d")
        self.panel_count_today    = self._load_daily_count()
        self.all_sequences_done   = False
        self._last_raw            = -1
        self._last_stable         = 0

        # ── Camera-2 OCR state (ADD ONLY — isolated from Camera-1) ────
        self.ocr_started          = False   # True once Camera2OCR.start_ocr() called
        self.partial_serial       = None    # live partial from cam2 OCR
        self.ocr_done             = False   # True once serial confirmed
        self.folder_renamed       = False   # True once folder is renamed to serial
        self.cam2_frame_saved     = False
        self.cam2_image_path      = None

        self.inference_engine, self.backend = load_inference_engine()

    def _load_daily_count(self):
        try:
            count_file = os.path.join(BASE_STORAGE, 'panel_count.txt')
            if os.path.exists(count_file):
                with open(count_file, 'r') as _f:
                    data = _f.read().strip().split(',')
                    if len(data) == 2 and data[0] == self.today_date:
                        return int(data[1])
        except Exception as e:
            print(f"[COUNT] Load error: {e}")
        return 0

    def refresh_daily_count(self):
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self.today_date:
            self.today_date        = new_date
            self.panel_count_today = 0
        else:
            self.panel_count_today = self._load_daily_count()

    def reset_sequence_timers(self):
        """Reset wiping timers and masks when moving to a new sequence."""
        self.cleaned_mask         = None
        self.last_wiping_time     = None
        self.wipe_t0              = None
        self.progress             = 0
        self.cleaning_percentage  = 0.0
        self.wiping_active        = False
        self.no_wiping_frames     = 0
        self.consecutive_wiping_frames = 0
        self.wipe_start_time      = None
        self.total_wipe_seconds   = 0.0
        self.hand_gone_since      = None
        # Reset per-sequence wiping counters
        self.seq1_wiping_frames   = 0
        self.seq2_wiping_frames   = 0
        self.seq3_wiping_frames   = 0
        self.seq3_no_wipe_frames  = 0
        self.seq3_completion_stage = 0
        # ── Auto-capture flags: NEVER reset on sequence transitions ─────────
        # seq2_auto_captured / seq3_auto_captured must survive SEQ2→SEQ3 so
        # the fallback blocks don't re-fire after capture is already done.
        # Only reset_for_new_panel() may clear these.
        self.seq1_snapshot_data   = None   # clear so SEQ1 re-takes snapshot next panel
        self.seq3_absence_frames  = 0
        self.seq3_seen_frames     = 0
        # ── Landscape counters: reset per sequence (not best frames) ─────────
        # seq2_best_frame / seq3_best_frame are kept so rescue fallbacks work.
        # Only the per-sequence COUNTERS reset so the new sequence starts fresh.
        self.seq2_landscape_count = 0
        self.seq2_miss_count      = 0
        self.seq2_prev_center     = None
        self.seq3_landscape_count = 0
        self.seq3_miss_count      = 0
        self.seq3_prev_center     = None

        # FIX 4: reset seq2_in_sequence_frames on sequence transition so the
        # 20-frame timeout does not fire immediately at SEQ2 start.
        self.seq2_in_sequence_frames = 0

    def reset_for_new_panel(self):

        """Reset all per-panel state when a new panel starts."""
        self.refresh_daily_count()
        self.current_sequence_seq = 0
        self.completed = {1: False, 2: False, 3: False}
        self.wipe_start = None
        self.progress = 0
        self.ocr_done = False
        self.serial = None
        self.current_sequence    = 0
        self.sequence_status     = {1: 'pending', 2: 'pending', 3: 'pending'}
        self.cleaned_mask        = None
        self.cleaning_percentage = 0.0
        self.failure_reason      = None
        self.last_panel_position = None
        self.serial_number       = "UNKNOWN"
        self.sequence_captured   = {1: False, 2: False, 3: False}
        self.panel_lost_count    = 0
        self.panel_absent_frames = 0
        self.wiping_active       = False
        self.ocr_candidates      = []
        self.all_sequences_done  = False
        self.panel_contour       = None
        self.panel_rect          = None
        self.panel_mask          = None
        self.hand_bbox           = None
        self.prev_gray           = None
        self.last_clean_frame      = None
        self.last_clean_panel_rect = None
        self.last_clean_contour    = None
        self.last_clean_serial_det = None
        self.seq_capture_data    = {1: None, 2: None, 3: None}
        self.detected_seq_history.clear()
        self.orig_frame = None
        self.seq1_clean_images.clear()
        self.seq1_snapshot_data = None
        self.seq3_absence_frames = 0
        self.seq3_seen_frames = 0
        self.real_wipe_seconds = 0.0
        self._seq2_countdown_time = None    # Reset countdown timer
        self._seq3_countdown_time = None    # Reset countdown timer
        self._seq2_prompt_time    = None
        self._seq3_prompt_time    = None
        self._seq2_prompted       = False
        self._seq3_prompted       = False
        self._seq2_cap_announced  = False
        self._seq3_cap_announced  = False
        # Reset lead times
        self.panel_start_time    = None
        self.seq_start_time      = {1: None, 2: None, 3: None}
        self.seq_end_time        = {1: None, 2: None, 3: None}
        self.panel_end_time      = None
        self.wipe_start_time     = None
        self.total_wipe_seconds  = 0.0
        self.hand_gone_since     = None
        self.last_wiping_time    = None
        self.current_sequence_panel_name = None
        self.current_sequence_panel_folder = None
        self.model_sees          = 'NONE'
        # Reset Camera-2 OCR flags for new panel
        self.ocr_started          = False
        self.ocr_done             = False
        self.partial_serial       = None
        self.folder_renamed       = False
        self.cam2_frame_saved     = False
        self.cam2_image_path      = None
        self.ocr_triggered_by_tilt = False   # reset portrait-tilt OCR flag
        if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
            camera2_ocr_instance.reset_for_new_panel()
        # Reset SEQ3 completion tracking
        self.seq1_wiping_frames   = 0
        self.seq2_wiping_frames   = 0
        self.seq3_wiping_frames   = 0
        self.seq3_no_wipe_frames  = 0
        self.seq3_completion_stage = 0
        self.seq3_wiping_started  = False
        self.seq3_consec_wipe_frames = 0  # strict consecutive counter — must reset per panel
        self.seq1_final_wipe_pct  = 0    # saved at SEQ1→SEQ2 transition
        self.seq2_final_wipe_pct  = 0    # saved at SEQ2→SEQ3 transition
        self.seq2_final_wiping_frames = 0 # saved at SEQ2→SEQ3 transition
        self.seq2_detection_confidence = 0   # confidence of SEQ2 when first detected
        self.seq3_final_wipe_pct  = 0    # saved at SEQ3 completion
        # ✅ FIX: Reset SEQ3 frame capture & tracking for new panel
        self.seq3_best_frame      = None  # Best captured frame for SEQ3
        self.seq3_best_sharp      = 0.0   # Sharpness of best frame
        self.seq3_best_meta       = {}    # Metadata for best frame
        self.seq3_auto_captured   = False # Has SEQ3 been auto-captured?
        self.seq3_landscape_count = 0     # Count of landscape frames
        self.seq3_miss_count      = 0     # Count of detection misses
        self.seq3_prev_center     = None  # Previous panel center (cx, cy)
        self.seq3_prev_center_x   = None  # Previous panel center X
        self.seq3_stable_count    = 0     # Stable frame counter for SEQ3
        self.seq3_absence_frames  = 0     # Frames with panel absent (already reset at 670)
        self.seq3_seen_frames     = 0     # Frames with panel present (already reset at 671)
        # ✅ FIX: Add SEQ2 20% wipe timing tracking
        self.seq2_20_percent_time = None   # When 20% wipe detected for SEQ2
        self.seq2_capture_triggered_at_pct = 0  # Progress % when SEQ2 capture fired
        self.current_sequence_panel_name = None
        self.last_wiping_time     = None
        # Reset auto-capture flags
        self.last_detected_seq      = 0
        self.seq1_auto_captured     = False
        self.seq2_auto_captured     = False
        self.seq3_auto_captured     = False
        # Reset stability counters for capture timing
        self.seq2_stable_count         = 0
        self.seq3_stable_count         = 0
        # NEW: reset per-zone hand-free delay
        self.seq_hand_free_since       = {1: None, 2: None, 3: None}
        self.seq_delay_triggered       = {1: False, 2: False, 3: False}
        # NEW: reset per-seq phase and post-capture wipe tracking
        self.seq_phase                 = {1: 'idle', 2: 'idle', 3: 'idle'}
        self.post_capture_wipe_secs    = {1: 0.0, 2: 0.0, 3: 0.0}
        self.post_capture_last_wipe    = {1: None, 2: None, 3: None}

        # Reset best landscape trackers for new panel
        self.seq2_best_frame      = None
        self.seq2_best_sharp      = 0.0
        self.seq2_best_meta       = {}
        self.seq2_landscape_count = 0
        self.seq2_miss_count      = 0
        self.seq2_prev_center     = None
        self.seq2_prev_center_x   = None  # Reset horizontal tracking
        self.seq3_best_frame      = None
        self.seq3_best_sharp      = 0.0
        self.seq3_best_meta       = {}
        self.seq3_landscape_count = 0
        self.seq3_miss_count      = 0
        self.seq3_prev_center     = None
        self.seq3_prev_center_x   = None  # Reset horizontal tracking
        self.seq3_absence_frames  = 0
        self.seq3_seen_frames     = 0
        # FIX H4: reset unconditional SEQ2 frame counter added by FIX H3
        self.seq2_in_sequence_frames   = 0
        # Reset saved wipe-% gates for the new panel
        self.seq2_final_wipe_pct  = 0
        self.seq3_final_wipe_pct  = 0
        # Clear first-frame snapshot so it is re-captured for each new panel
        self.seq1_first_clean_frame = None
        # FIX F1: reset attributes that are set dynamically and not listed above.
        # Without these, fallback captures use OLD panel images after a reset.
        self.seq1_detection_count   = 0    # counter for SEQ1 stability
        self.seq2_stable_for_seq1   = 0    # consecutive frames SEQ2 active (guards early SEQ1 completion)
        self.last_clean             = None # last hand-free frame (process_frame)
        self.orig                   = None # raw current frame (process_frame)
        self.was_absent_long        = False

        self.panel_bbox_xyxy        = None   # FIX 3: reset per-panel panel bbox
        self.panel_id            = None   # DEFERRED — generated when first image captured
        _grabcut_cache.clear()
        self.panel_reset_id += 1
        # Reset landscape alert for new panel
        self.landscape_alert     = ""
        self.landscape_alert_ts  = 0.0
        self._seq2_cap_announced = False
        self._seq3_cap_announced = False
        self._seq2_prompted      = False
        self._seq3_prompted      = False
        self._seq2_prompt_time   = None
        self._seq3_prompt_time   = None
        # [SEQ2 COMPLETION GATE] new tracking vars — reset each panel
        self.seq2_capture_time         = None  # time.time() when SEQ2 image first saved
        self.seq2_post_capture_wipe_pct = 0    # progress % at capture moment
        self.seq2_post_cap_seq3_frames  = 0    # SEQ3 wipe frames counted after SEQ2 captured
        self.seq3_capture_time          = None
        self.seq3_post_capture_wipe_pct = 0
        self.seq3_post_cap_seq3_frames  = 0
        self._seq2_post_cap_wipe_frames = 0
        self._seq3_post_cap_wipe_frames = 0
        self._missed_alert_fired        = None
        print(f"🔄 New panel — all sequences reset (reset_id={self.panel_reset_id})")


state = PanelDetectionState()


# ─────────────────────────────────────────────────────────────
# PANEL DETECTION  (OpenCV — untouched)
# ─────────────────────────────────────────────────────────────

_grabcut_cache = {}   # {bbox_key: contour}

def detect_panel_contour_in_bbox(frame, bbox):
    """
    Returns a tight rectangular polygon based on the bbox.
    (Disabled GrabCut due to extreme CPU overhead on Raspberry Pi when wiping).
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)
    
    return np.array([[[x1,y1]],[[x2,y1]],[[x2,y2]],[[x1,y2]]], dtype=np.int32)


# ─────────────────────────────────────────────────────────────
# HAND DETECTION  (calls active engine)
# ─────────────────────────────────────────────────────────────

def detect_hand(frame):
    """Refactored to generic detect_objects for all classes."""
    if state.inference_engine is None:
        return []
    try:
        # Lowered threshold to 0.15 for better sensitivity on CPU
        return state.inference_engine.infer(frame, conf_threshold=0.15)
    except Exception as e:
        print(f"Inference error: {e}")
    return []


# ─────────────────────────────────────────────────────────────
# MOTION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_motion(frame, prev_gray):
    # Downscale to a small working resolution before any expensive operations.
    # At 1280×960, a 21×21 GaussianBlur takes ~100-150 ms on Pi 4B.
    # At 320×240 it takes ~6 ms — 25× faster.
    # detect_wiping() already resizes its inputs to (640,480) so the smaller
    # mask is fine: it just gets upscaled there instead of downscaled.
    MOTION_W, MOTION_H = 320, 240
    small = cv2.resize(frame, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (9, 9), 0)   # smaller kernel for smaller frame
    if prev_gray is None:
        return gray, None
    # Shape mismatch guard: reconnect may change resolution mid-session
    if prev_gray.shape != gray.shape:
        return gray, None
    delta  = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    return gray, thresh


# ─────────────────────────────────────────────────────────────
# WIPING DETECTION  (small kernel — gradual progress)
# ─────────────────────────────────────────────────────────────

def detect_wiping(hand_bbox, motion_mask, panel_mask):
    """
    On CPU, motion might be jittery. If we have a hand inside the panel,
    we count it as wiping even if motion is subtle.
    """
    if hand_bbox is None or panel_mask is None:
        return None
    
    target_size = (640, 480)
    try:
        # Create hand mask
        hand_mask = np.zeros_like(panel_mask)
        x1, y1, x2, y2 = hand_bbox
        hand_mask[y1:y2, x1:x2] = 255
        
        # Resize both to 640x480
        p_resized = cv2.resize(panel_mask, target_size)
        h_resized = cv2.resize(hand_mask, target_size)
        
        # Core wiping: Intersection of Hand and Panel
        wiping = cv2.bitwise_and(h_resized, p_resized)
        
        # If motion mask exists, use it to refine
        if motion_mask is not None and motion_mask.size > 0:
            m_dilated = cv2.dilate(motion_mask, np.ones((15, 15), np.uint8))
            if m_dilated.shape[1] != target_size[0] or m_dilated.shape[0] != target_size[1]:
                m_dilated = cv2.resize(m_dilated, target_size)
            wiping = cv2.bitwise_and(wiping, m_dilated)
    except Exception as e:
        # print(f"[DEBUG] detect_wiping error: {e}")
        return 0.0
    
    # Expand the 'cleaned' area significantly around the hand
    wiping = cv2.dilate(wiping, np.ones((11, 11), np.uint8), iterations=1)
    return wiping


# ─────────────────────────────────────────────────────────────
# BLUE TINT
# ─────────────────────────────────────────────────────────────

def apply_blue_tint(frame, panel_mask, cleaned_mask):
    """Removed blue tint overlays: return the original frame unchanged.

    The panel mask/cleaned mask highlighting has been disabled per request.
    """
    return frame


# ─────────────────────────────────────────────────────────────
# CLEANING PERCENTAGE  (smoothed — max +2 % per frame)
# ─────────────────────────────────────────────────────────────

def calculate_cleaning_percentage(panel_mask, cleaned_mask):
    """Returns the raw % of the panel area covered by cleaned_mask.
    No per-frame cap — updates instantly so the progress bar reflects
    actual wipe coverage without lag.
    """
    if panel_mask is None or cleaned_mask is None:
        return 0.0
    panel_area   = cv2.countNonZero(panel_mask)
    if panel_area == 0:
        return 0.0
    cleaned_area = cv2.countNonZero(cv2.bitwise_and(panel_mask, cleaned_mask))
    return round((cleaned_area / panel_area) * 100.0, 1)


# ─────────────────────────────────────────────────────────────
# PANEL FLIP DETECTION
# ─────────────────────────────────────────────────────────────

def detect_panel_flip(is_panel_present, stable_model_seq):
    """
    Panel continuity logic:
    - Same panel if continuous or short gap, even if rotated.
    - Flip/New Panel ONLY if absent for > 60 frames AND a new panel (SEQ1) appears.
    """
    if not is_panel_present:
        state.panel_absent_frames += 1
        return False
    
    # Panel is present — reset absence counter
    absent = state.panel_absent_frames
    state.panel_absent_frames = 0
    
    # Protect against None (not enough history yet)
    if stable_model_seq is None:
        return False
    
    # If the panel was gone for > 60 frames (~2.4s at 25fps)
    # AND the model now detects a fresh SEQ1 panel, it's a completely new panel.
    # Otherwise, it's the same panel being rotated or repositioned.
    if absent > 60 and stable_model_seq == 1 and state.current_sequence > 0:
        print(f"🔄 New panel introduced (was absent for {absent} frames)")
        return True
        
    return False


# ─────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────

def crop_serial_roi_dynamic(frame, detections):
    for det in detections:
        if 'serial' in det['name'].lower():
            x1, y1, x2, y2 = det['bbox']

            # Add padding
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            return frame[y1:y2, x1:x2]

    return None

def crop_serial_roi_fallback(frame):
    h, w = frame.shape[:2]
    return frame[int(h*0.1):int(h*0.25), int(w*0.05):int(w*0.35)]

# OCR logic removed per request





# ─────────────────────────────────────────────────────────────
# FOLDER HELPER
# ─────────────────────────────────────────────────────────────

def get_panel_folder(serial_number, panel_id=None):
    """Returns a unique, stable folder path for the current panel session."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    if not panel_id:
        panel_id = datetime.now().strftime("%H%M%S")
    
    # Use a unique ID-based folder name instead of renaming based on serial
    # This prevents the "Missing Images" bug in PDF generation.
    folder_name = f"SSP-SEQ_{panel_id}"
    folder = os.path.join(BASE_STORAGE, date_str, folder_name)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        if e.errno == 28:
            print(f"\n[CRITICAL] ❌ DISK FULL: Cannot create session folder. Please clear space on the device!")
        else:
            print(f"\n[ERROR] Folder creation failed: {e}")
        return None
    return folder


def ensure_panel_folder():
    """Returns the stable folder for the current panel session.

    Creates a new folder the first time it is called for a panel.
    Guards against creating a premature/orphaned folder by requiring
    that at least one sequence is active (current_sequence > 0) or
    that a panel_id is already set from this cycle.

    The orphaned-folder problem (SEQ2/SEQ3 detected before SEQ1) is
    prevented in update_seq() which blocks sequence advancement unless
    the previous sequence was genuinely active first.
    """
    # FIX 2: removed seq1_snapshot_data guard — snapshot never committed
    # because Camera 1 serial_det is always None (wrong face for Camera 1).
    # Only block before SEQ1 starts (current_sequence < 1).
    if not state.panel_id:
        if state.current_sequence < 1:
            return state.current_sequence_panel_folder
        state.panel_id = datetime.now().strftime("%H%M%S")

    if state.current_sequence_panel_folder is None:
        state.current_sequence_panel_folder = get_panel_folder(
            state.serial_number, state.panel_id)

    # FIX 4: if stored folder no longer exists on disk (was renamed or deleted)
    # sync from camera2_ocr which always has the latest renamed path.
    _p = state.current_sequence_panel_folder
    if _p and not os.path.exists(_p):
        if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
            _c2p = getattr(camera2_ocr_instance, 'panel_folder', None)
            if _c2p and _c2p not in ('.', '') and os.path.exists(_c2p):
                state.current_sequence_panel_folder = _c2p
        # If still doesn't exist, recreate it
        if not os.path.exists(state.current_sequence_panel_folder or ''):
            os.makedirs(state.current_sequence_panel_folder, exist_ok=True)

    return state.current_sequence_panel_folder

def allocate_panel_folder_for_seq1(serial_present=False):
    """Create the panel save folder as soon as SEQ1 is stable."""
    if state.current_sequence_panel_folder is not None:
        return state.current_sequence_panel_folder

    if (state.seq1_detection_count >= 2
            and state.current_sequence in (0, 1)):
        # FIX 1: only generate panel_id if not already set
        # Was: always overwrote panel_id → two folders with different timestamps
        if not state.panel_id:
            state.panel_id = datetime.now().strftime("%H%M%S")
        state.current_sequence_panel_folder = get_panel_folder(
            state.serial_number or 'SSP-SEQ', state.panel_id)
        print(f"[SEQ1-FOLDER] ✅ Panel folder created (EARLY - 2 stable frames)")
        print(f"[SEQ1-FOLDER]    Path: {state.current_sequence_panel_folder}")
        print(f"[SEQ1-FOLDER]    Panel ID: {state.panel_id}")
        print(f"[SEQ1-FOLDER]    Detection count: {state.seq1_detection_count}")
        print(f"[SEQ1-FOLDER]    Current sequence: {state.current_sequence}")
        if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
            camera2_ocr_instance.set_panel_folder(state.current_sequence_panel_folder)
            print(f"[SEQ1-FOLDER]    Camera-2 folder assigned")

    return state.current_sequence_panel_folder



# ─────────────────────────────────────────────────────────────
# IMAGE CAPTURE
#
# SEQ1  →  full frame  +  left / middle / right / bottom crops
# SEQ2  →  full frame only
# SEQ3  →  full frame only
# SEQ4  →  full frame  +  ROI serial number (rotated 90° CCW)
# ─────────────────────────────────────────────────────────────

def rename_panel_images(folder, serial):
    """
    Rename all files saved with the placeholder prefix "PANEL" to the real
    serial number.  Recurses into the 'crops' subfolder too.
    Also renames OCR_Frame files so the serial is traceable.

    Before: PANEL_Seq1_Full_193145.jpg  /  OCR_Frame01_v1_193145.jpg
    After:  060526099A_Seq1_Full_193145.jpg  /  060526099A_OCR_Frame01_v1_193145.jpg
    """
    if not folder or not os.path.isdir(folder):
        print(f"  [RENAME] Skipped — invalid folder: {folder}")
        return

    if not serial or serial in ("Searching...", "UNKNOWN", "Reading...", "SSP-SEQ", ""):
        return

    safe_serial = re.sub(r'[^A-Za-z0-9_-]', '', str(serial))
    if not safe_serial:
        return

    import glob
    # Include crops subfolder
    search_dirs = [folder, os.path.join(folder, 'crops')]

    renamed_count = 0
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        # Rename SSP-SEQ_ prefixed files
        for old_path in glob.glob(os.path.join(search_dir, "SSP-SEQ_*.*")):
            basename = os.path.basename(old_path)
            new_basename = safe_serial + basename[len("SSP-SEQ"):]   # "SSP-SEQ_Seq1..." → "060526099A_Seq1..."
            new_path = os.path.join(search_dir, new_basename)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  [RENAME] {basename} → {new_basename}")
                    renamed_count += 1
            except Exception as e:
                print(f"  [RENAME ERR] {basename}: {e}")

        # Rename legacy PANEL_ prefixed files
        for old_path in glob.glob(os.path.join(search_dir, "PANEL_*.*")):
            basename = os.path.basename(old_path)
            new_basename = safe_serial + basename[len("PANEL"):]   # "PANEL_Seq1..." → "060526099A_Seq1..."
            new_path = os.path.join(search_dir, new_basename)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  [RENAME] {basename} → {new_basename}")
                    renamed_count += 1
            except Exception as e:
                print(f"  [RENAME ERR] {basename}: {e}")

        # Rename OCR_Frame files — add serial prefix so they are traceable
        for old_path in glob.glob(os.path.join(search_dir, "OCR_Frame*.jpg")):
            basename = os.path.basename(old_path)
            new_basename = f"{safe_serial}_{basename}"
            new_path = os.path.join(search_dir, new_basename)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  [RENAME] {basename} → {new_basename}")
                    renamed_count += 1
            except Exception as e:
                print(f"  [RENAME ERR] {basename}: {e}")

    if renamed_count:
        print(f"[RENAME] {renamed_count} files renamed to serial '{safe_serial}'")

def _save(path, img):
    """Saves image with high quality for PDF reports."""
    if img is None or img.size == 0:
        print(f"  ❌ Cannot save empty image to {os.path.basename(path)}")
        return False
    # High quality JPEG (98) for QC checks — better than standard 90
    ok = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"  {'💾' if ok else '❌'} {os.path.basename(path)}")
    return ok


def safe_serial_filename(serial):
    """
    Sanitize serial for use in filenames.
    ANY placeholder value → "SSP-SEQ" so rename_panel_images can find files by
    a single consistent prefix rather than trying to match 'Searching', 'UNKNOWN' etc.
    """
    _PLACEHOLDERS = {None, '', 'UNKNOWN', 'Searching...', 'Searching',
                     'Reading...', 'Reading', 'PANEL', 'SSP-SEQ', 'INVALID FRAME'}
    if serial in _PLACEHOLDERS:
        return "SSP-SEQ"
    s = re.sub(r'[^A-Za-z0-9_-]', '', str(serial))
    return s if s else "SSP-SEQ"



def capture_sequence_images(cap_data, sequence_id, serial_number):
    """
    Save images for a sequence at full camera resolution (JPEG quality=98).

    SEQ1 → Full landscape frame  +  Left / Middle / Right / Bottom crops
    SEQ2 → Full landscape frame only
    SEQ3 → Full landscape frame only

    CROP STRATEGY (SEQ1):
    Crops are taken from the FULL LANDSCAPE FRAME, NOT from the YOLO bbox.
    Reason: YOLO bbox can be portrait when panel is tilted → rotating the bbox
    crop makes Left/Middle/Right crops appear sideways.  The panel fills ~80%
    of the frame in the reference layout, so full-frame percentage crops give
    consistent, well-oriented views every time.

    Reference frame (CP Plus 1280×960 horizontal):
    ┌──────────────────────────────────────────┐
    │  Left  │     Middle      │    Right      │
    │  0-45% │   28%-72%       │  55%-100%     │
    │                                          │
    │        Bottom (55%-100% height)          │
    └──────────────────────────────────────────┘

    Filename prefix:
      Serial confirmed → {serial}_Seq{N}_Full_{ts}.jpg
      Serial pending   → PANEL_Seq{N}_Full_{ts}.jpg
    """
    try:
        folder = ensure_panel_folder()
        if folder is None:
            # Between-panel window: panel session not yet active.
            # Do NOT create a folder here — the capture is premature.
            print(f"  ❌ SEQ{sequence_id}: no active panel folder (between panels) — skipped")
            return None
        ts     = datetime.now().strftime("%H%M%S")
        safe_s = safe_serial_filename(serial_number)
        saved  = {}
        print(f"\n📸 Capturing SEQ{sequence_id} — serial='{safe_s}' → {folder}")

        frame = cap_data['frame']
        if frame is None or frame.size == 0:
            print(f"  ❌ SEQ{sequence_id}: frame is None/empty — skipped")
            return None

        fh, fw = frame.shape[:2]
        print(f"  Frame size: {fw}×{fh} px")

        # ── Step 1: Ensure landscape orientation ─────────────────────────
        # Camera is normally 1280×960 (landscape).  Guard against portrait
        # RTSP sources or cameras mounted on their side.
        # FIX D7: >= catches square frames (e.g. 640x640) too
        if fh >= fw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            fh, fw = frame.shape[:2]
            print(f"  Rotated to landscape: {fw}x{fh}")

        # Report sharpness so we can track frame quality in logs
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # ── SEQ2/SEQ3: Always save the FULL horizontal camera frame ──────
        # STRICT: no bbox crop. Full landscape frame only.
        # Step 1 above already ensured landscape orientation.
        # A frame that is still portrait here means Step-1 failed (very unusual).
        if sequence_id in (2, 3):
            # Re-evaluate after possible rotation in Step 1
            if fh > fw:
                # Attempt a second rotation as last resort
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                fh, fw = frame.shape[:2]
                print(f"  ⚠️  SEQ{sequence_id}: secondary rotation applied → {fw}×{fh}")
            if fh > fw:
                print(f"  ❌ SEQ{sequence_id}: frame still NOT landscape "
                      f"({fw}×{fh}) after rotation — skipped")
                return None
            full_path = os.path.join(folder, f"{safe_s}_Seq{sequence_id}_Full_{ts}.jpg")
            if _save(full_path, frame):
                saved['main'] = full_path
                size_kb = os.path.getsize(full_path) // 1024
                print(f"  ✅ SEQ{sequence_id} FullFrame: {os.path.basename(full_path)}"
                      f"  ({fw}×{fh}, {size_kb} KB, sharpness={sharpness:.0f})")
            else:
                print(f"  ❌ SEQ{sequence_id} full-frame save FAILED")
                return None
            return saved

        # ── Step 2: Save full-resolution landscape frame (SEQ1 only) ─────
        full_path = os.path.join(folder, f"{safe_s}_Seq{sequence_id}_Full_{ts}.jpg")
        if _save(full_path, frame):
            saved['main'] = full_path
            size_kb = os.path.getsize(full_path) // 1024
            print(f"  ✅ SEQ{sequence_id} Full: {os.path.basename(full_path)}"
                  f"  ({fw}×{fh}, {size_kb} KB, sharpness={sharpness:.0f})")
        else:
            print(f"  ❌ SEQ{sequence_id} Full save FAILED")
            return None

        # ── Step 3: REMOVED (User requested not to store the 4 crops) ──
        return saved

    except Exception as e:
        import traceback
        print(f"❌ Capture error SEQ{sequence_id}: {e}")
        traceback.print_exc()
        return None


def check_missing_sequences():
    """
    Check if any sequence images are missing before PDF generation.
    Alert user with voice + UI message if sequences are incomplete.
    Returns True if all sequences captured, False if any missing.
    """
    missing = []
    for seq_id in (1, 2, 3):
        if not state.sequence_captured.get(seq_id, False):
            missing.append(seq_id)
    
    if missing:
        missing_str = ", ".join(f"SEQ{s}" for s in missing)
        state.status_msg = f"⚠️  MISSING: {missing_str} — Please capture before PDF"
        
        # Voice alert in English and Hindi
        msg_en = f"Missing {missing_str}. Please place the panel and capture images."
        speak(msg_en, lang='en')
        
        def _hi_missing():
            import shutil
            seq_hi = {1: 'ek', 2: 'do', 3: 'teen'}
            missing_hi = ", ".join(f"SEQ {seq_hi.get(s, str(s))}" for s in missing)
            msg_hi = f"{missing_hi} missing. Panel rakhein aur capture karein."
            exe = ('espeak-ng' if shutil.which('espeak-ng')
                   else 'espeak' if shutil.which('espeak') else None)
            if exe:
                try:
                    subprocess.run([exe, '-v', 'hi', '-s', '120', '-a', '90', msg_hi],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
                except Exception:
                    pass
        
        threading.Thread(target=_hi_missing, daemon=True).start()
        return False
    
    return True


def check_panel_wipe_status():
    """
    Check if panel was properly wiped for each sequence.
    Alert user if wipe percentage is too low.
    Returns wipe status for logging/reporting.
    """
    wipe_status = {}
    MIN_WIPE_PCT = 50  # Minimum acceptable wipe percentage
    
    for seq_id in (1, 2, 3):
        if seq_id == 1:
            wipe_pct = getattr(state, 'seq1_final_wipe_pct', 0)
        elif seq_id == 2:
            wipe_pct = getattr(state, 'seq2_final_wipe_pct', 0)
        else:
            wipe_pct = getattr(state, 'seq3_final_wipe_pct', 0)
        
        wipe_status[seq_id] = wipe_pct
        
        if wipe_pct < MIN_WIPE_PCT:
            # Alert operator about insufficient wiping
            state.status_msg = f"⚠️  SEQ{seq_id}: Panel not properly wiped ({wipe_pct:.0f}%)"
            
            msg_en = f"Sequence {seq_id}. Panel not properly wiped. Please wipe clean."
            speak(msg_en, lang='en')
            
            def _hi_wipe(sid=seq_id):
                import shutil
                seq_hi = {1: 'ek', 2: 'do', 3: 'teen'}.get(sid, str(sid))
                msg_hi = f"Sequence {seq_hi}. Panel saf nahi hai. Saaf karein."
                exe = ('espeak-ng' if shutil.which('espeak-ng')
                       else 'espeak' if shutil.which('espeak') else None)
                if exe:
                    try:
                        subprocess.run([exe, '-v', 'hi', '-s', '120', '-a', '90', msg_hi],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
                    except Exception:
                        pass
            
            threading.Thread(target=_hi_wipe, daemon=True).start()
    
    return wipe_status


def _finalize_panel():
    """All 3 sequences done — increment count and fire PDF in background."""
    # [FIX] Capture the current folder BEFORE any state resets happen
    # This prevents the "PDF in next panel folder" bug.
    folder = state.current_sequence_panel_folder
    if folder is None:
        folder = ensure_panel_folder()
    
    if folder is None:
        print("⚠️ [PDF] Cannot generate PDF: No panel folder available (no images captured).")
        return

    if state.all_sequences_done:
        return
    state.all_sequences_done  = True
    state.panel_count_today  += 1
    # Persist panel count to file so it survives restarts
    try:
        count_file = os.path.join(BASE_STORAGE, 'panel_count.txt')
        os.makedirs(BASE_STORAGE, exist_ok=True)
        with open(count_file, 'w') as _f:
            _f.write(f"{state.today_date},{state.panel_count_today}")
    except Exception as _ce:
        print(f"[COUNT] Save error: {_ce}")
    state.current_sequence    = 0
    state.panel_end_time      = time.time()

    # Lock all statuses — also set completed[i]=True so the /api/sequence_status
    # normalization loop doesn't overwrite these back to 'pending' on the next poll.
    for i in (1, 2, 3):
        if state.sequence_status.get(i) not in ('completed', 'missed'):
            state.sequence_status[i] = 'completed'
        # Sync completed flag — normalization checks this first; without it the
        # else branch would reset status to 'pending' every 1-second poll cycle.
        if state.sequence_status.get(i) == 'completed':
            state.completed[i] = True
        # Guard: only set seq_end_time if NOT already frozen by a mid-cycle event
        # (e.g. SEQ2 end time is frozen when SEQ3 activates — don't overwrite with panel_end_time)
        if state.seq_end_time.get(i) is None:
            state.seq_end_time[i] = state.panel_end_time

    # Calculate lead times
    panel_start = state.panel_start_time or state.panel_end_time
    total_secs  = state.panel_end_time - panel_start

    seq_times = {}
    for i in (1, 2, 3):
        s = state.seq_start_time.get(i)
        e = state.seq_end_time.get(i)
        if s and e:
            seq_times[i] = round(e - s, 1)
        else:
            seq_times[i] = None

    print("\n" + "="*60)
    print(f"🎉  PANEL {state.panel_count_today} COMPLETE — ALL SEQUENCES DONE")
    print(f"⏱️  TOTAL CUMULATIVE LEAD TIME: {total_secs:.1f}s")
    print("-" * 60)
    for i in (1, 2, 3):
        t = seq_times.get(i)
        print(f"    SEQ{i} Duration: {t:.1f}s" if t else f"    SEQ{i} Duration: N/A")
    print("="*60 + "\n")

    # Using the 'folder' path captured at the very beginning of finalization
    
    if state.serial_number:
        pdf_serial = state.serial_number
    else:
        pdf_serial = "UNKNOWN"
        
    serial_snap = pdf_serial
    seq_times_snap   = dict(seq_times)
    total_secs_snap  = total_secs
    panel_start_snap = panel_start

    if state.serial_number == "UNKNOWN":
        print("⚠️ Serial not detected properly")

    cam2_image = getattr(state, 'cam2_image_path', None)

    def _bg():
        try:
            # Snapshot reset_id at start — used to abort state writes if a
            # new panel cycle started (reset_panel was called) before this
            # PDF thread finishes. Without this guard, _bg() writing
            # state.serial_number after a reset overrides "Searching..." and
            # the UI shows the OLD panel's serial again (Bug 1).
            my_reset_id = state.panel_reset_id

            # ── Check for missing sequences before PDF generation ──
            # Alert user with voice + UI if any sequence images are missing
            if not check_missing_sequences():
                print("[PDF] ⚠️  Incomplete sequence capture — PDF generation deferred")
                print("      User must complete all sequence captures before PDF")
                return

            # ── Check panel wipe status and alert if insufficient ──
            wipe_status = check_panel_wipe_status()
            print(f"[PDF] Wipe status: {wipe_status}")

            # ── FIX: wait up to 8s for background OCR to produce a real serial ──
            final_serial = serial_snap
            if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
                t0 = time.time()
                while time.time() - t0 < 8.0:
                    s = camera2_ocr_instance.get_serial_number()
                    if s and s not in (None, 'UNKNOWN', 'Reading...', 'Searching...', ''):
                        final_serial = s
                        # Only update state if this panel hasn't been reset yet
                        if state.panel_reset_id == my_reset_id:
                            if state.serial_number in (None, 'UNKNOWN', 'Searching...', 'Reading...', ''):
                                state.serial_number = s
                        print(f"[PDF] OCR result ready in {time.time()-t0:.1f}s → serial={s}")
                        break
                    time.sleep(0.25)
                else:
                    print(f"[PDF] OCR not ready after 8s — using serial='{final_serial}'")
            # ── Rename all images with confirmed serial BEFORE generating PDF ─
            # This ensures every file in the folder carries the right serial
            # prefix and the PDF itself is stored in the serial-named folder.
            # FIX A1: always use the folder captured at _finalize_panel() entry.
            # state.current_sequence_panel_folder may point to the NEXT panel's
            # folder by the time _bg() runs (up to 8 s later for OCR wait).
            pdf_folder = folder
            if final_serial not in (None, 'UNKNOWN', 'Searching...', ''):
                rename_panel_images(pdf_folder, final_serial)
                print(f"[PDF] All images renamed → serial='{final_serial}'")

            # FIX F2: robustly locate the serial/cam2 image.
            # 1. Start with the early-captured local variable.
            # 2. Re-read state only if it still belongs to this panel (reset_id guard).
            # 3. Always fall back to a folder scan so reset_panel() can't lose the image.
            final_cam2_image = cam2_image  # early capture from _finalize_panel() start
            if state.panel_reset_id == my_reset_id:
                live = getattr(state, 'cam2_image_path', None)
                if live and os.path.exists(live):
                    final_cam2_image = live
            # Folder scan — finds Cam2_ROI, Cam2_FullFrame, Serial_Original, OCR_Frame
            if not final_cam2_image or not os.path.exists(str(final_cam2_image)):
                import glob as _gl
                for _pat in [
                    f"{final_serial}_Cam2_ROI_*.jpg",
                    f"{final_serial}_Cam2_FullFrame_*.jpg",
                    f"{final_serial}_Serial_Original.jpg",
                    "*_Cam2_ROI_*.jpg",
                    "*_Serial_Original.jpg",
                ]:
                    _hits = sorted(_gl.glob(os.path.join(pdf_folder, _pat)))
                    if _hits:
                        final_cam2_image = _hits[-1]
                        print(f'[PDF] Found serial image: {os.path.basename(final_cam2_image)}')
                        break
            if final_serial not in (None, 'UNKNOWN', 'Searching...', ''):
                import glob as _gl
                _ocr_hits = sorted(_gl.glob(
                    os.path.join(pdf_folder, f"{final_serial}_OCR_Frame*.jpg")))
                if _ocr_hits:
                    final_cam2_image = _ocr_hits[-1]

            print(f"[DEBUG PDF] 🚀 Generating PDF → {os.path.basename(pdf_folder) if pdf_folder else 'None'}")
            print(f"  - Serial: {final_serial}")
            print(f"  - Folder: {pdf_folder}")
            print(f"  - Cam2 Image: {final_cam2_image}")

            pdf_path = generate_pdf_report(
                final_serial,
                pdf_folder,
                seq_times=seq_times_snap,
                total_time=total_secs_snap,
                panel_start=panel_start_snap,
                cam2_image=final_cam2_image
            )
            
            # --- FINAL DIAGNOSTICS LOG ---
            import glob as _gl
            saved_images = len(_gl.glob(os.path.join(pdf_folder, '*.jpg'))) + len(_gl.glob(os.path.join(pdf_folder, 'crops', '*.jpg')))
            log.panel_end(
                serial=final_serial,
                result="SUCCESS" if pdf_path else "FAILED",
                images_saved=saved_images,
                pdf_path=pdf_path or "",
                failure_reason=getattr(state, 'failure_reason', '') if not pdf_path else "",
                seq_durations=seq_times_snap,
                capture_statuses=getattr(state, 'seq_capture_status', {})
            )
            
            if pdf_path:
                print(f"[DEBUG PDF] ✅ PDF generated successfully at: {pdf_path}")
            else:
                print("[DEBUG PDF] ⚠️  PDF generation returned None")
                state.failure_reason = "PDF Generation Failed: Missing images"
        except Exception as exc:
            import traceback
            err_msg = f"PDF Error: {str(exc)}"
            print(f"⚠️  {err_msg}")
            state.failure_reason = err_msg
            traceback.print_exc()

    threading.Thread(target=_bg, daemon=True).start()

    # Reset UI after a short delay so operator sees "COMPLETED" briefly.
    # Guard with reset_id snapshot: if a new panel arrives in <2s,
    # process_frame already called reset_panel() — don't reset again or
    # it will clear the new panel's serial mid-detection (Bug 3).
    _reset_id_at_finalize = state.panel_reset_id
    def _delayed_reset():
        time.sleep(2.0)
        if state.panel_reset_id == _reset_id_at_finalize:
            reset_panel()
        else:
            print("[RESET] Delayed reset skipped — new panel already started")
    threading.Thread(target=_delayed_reset, daemon=True).start()


def advance_sequence():
    """Manual advance for UI button"""
    if state.current_sequence < 3:
        update_seq(state.current_sequence + 1)
    else:
        complete(3)
    return True

    # SEQ3 done → full panel complete — route through _finalize_panel
    # which handles count increment + PDF + cleanup
    _finalize_panel()
    return False

def cleanup_panel_images(folder):
    """
    After PDF is confirmed saved, remove all .jpg images from the panel folder
    and its crops/ subfolder. Keeps the PDF. Never touches the _uploads folder.
    """
    import glob as _glob
    try:
        removed = 0
        # Remove JPGs in panel folder
        for jpg in _glob.glob(os.path.join(folder, '*.jpg')):
            try:
                os.remove(jpg)
                removed += 1
            except Exception:
                pass
        # Remove crops subfolder entirely
        crops_dir = os.path.join(folder, 'crops')
        if os.path.isdir(crops_dir):
            import shutil
            shutil.rmtree(crops_dir, ignore_errors=True)
        print(f"  [CLEANUP] Removed {removed} jpg(s) + crops/ from {folder}")
    except Exception as e:
        print(f"  [WARN] Cleanup failed: {e}")


# check_sequence_completion removed — completion is now driven
# exclusively by the sequence state machine in process_frame.


# ─────────────────────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────
# SEQUENCE HELPERS
# ─────────────────────────────────────────────────────────────




# Duplicate sequence capture and finalization helpers removed

# ─────────────────────────────────────────────────────────────
# USER REQUESTED LOGIC
# ─────────────────────────────────────────────────────────────

def filter_detections(detections, frame):
    """
    Strict Filtering (User Requirement):
    1. Min confidence 0.6 (0.5 for hand)
    2. Only ONE detection per class
    3. Sequence-aware selection (Priority to current sequence)
    4. 70% Area Filter
    """
    # ── 5. DEBUG LOGGING (MANDATORY) ──
    print("\n--- MODEL DETECTIONS ---")
    if not detections:
        print("None")
    else:
        for d in detections:
            print(f"{d['name']} | conf={d['confidence']:.2f}")

    if not detections:
        return []
        
    best_per_class = {}
    for d in detections:
        name = d['name']
        conf = d.get('confidence', 0.0)
        
        # Normalization
        if name == "serial_number_region":
            name = "serial_number"
            d['name'] = "serial_number"
            
        # Confidence floors (Lowered hand threshold to 0.35 to be very aggressive)
        thresh = 0.35 if 'hand' in name else 0.6
        if conf < thresh:
            continue
            
        if name not in best_per_class or conf > best_per_class[name]['confidence']:
            best_per_class[name] = d
            
    # ── 2. Sequence-aware selection (STRICT) ──
    seq_candidates = [v for k, v in best_per_class.items() if k in CLASS_MAP]
    other_dets = [v for k, v in best_per_class.items() if k not in CLASS_MAP]
    
    # [RULE] Serial Number Presence = SEQ1
    has_serial = any(d['name'] == 'serial_number' for d in other_dets)
    
    panel_det = None
    if seq_candidates:
        # Expected ID: current active one, or 1 if just starting or if serial is detected
        expected_id = state.current_sequence
        if expected_id == 0:
            expected_id = 1
        
        # If we see a serial number, we MUST be looking for SEQ1
        if has_serial:
            expected_id = 1
            
        expected_name = f"panel_seq{expected_id}"

        # Priority 1: exact match with expected sequence
        for d in seq_candidates:
            if d['name'] == expected_name:
                panel_det = d
                break

        # Priority 2: fallback to highest confidence
        if panel_det is None:
            panel_det = max(seq_candidates, key=lambda x: x['confidence'])

    # ── 3. Strict sequence validation ──
    if panel_det is not None:
        det_id = CLASS_MAP.get(panel_det['name'], 0)
        
        # Determine what we expect (same logic as selection)
        expected_id = state.current_sequence if state.current_sequence > 0 else 1
        if has_serial:
            expected_id = 1
            
        if det_id != expected_id:
            print(f"[FILTER] Ignored wrong sequence {det_id}, expected {expected_id}")
            panel_det = None

    # ── 4. Area-based false detection filter ──
    if panel_det is not None:
        x1, y1, x2, y2 = panel_det['bbox']
        area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        if area > 0.7 * frame_area:
            print("[FILTER] Ignored large false detection")
            panel_det = None
        
    return ([panel_det] if panel_det else []) + other_dets

def get_seq(dets):
    """
    Strict sequence selection: selects the HIGHEST confidence sequence detection.
    Guarantees no order-dependency.
    
    NOTE: If we are at the start (IDLE), we FAVOR SEQ1 if detected to prevent 
    model bias from skipping straight to SEQ2.
    """
    if not dets:
        state.model_sees = "NONE"
        return None

    # ONLY sequence detections (strict mapping)
    seq_dets = [d for d in dets if d['name'] in CLASS_MAP]

    if not seq_dets:
        state.model_sees = "NONE"
        return None

    # [FIX] Require minimum confidence for ALL sequences to prevent false flips and ghost panels
    # ✅ ENHANCED: Increased threshold 0.60→0.70 to eliminate empty table false positives
    # Table textures rarely exceed 0.65 confidence; real panels typically 0.75+
    seq_dets = [d for d in seq_dets if d['confidence'] >= 0.70]
    
    if not seq_dets:
        state.model_sees = "NONE"
        return None

    # [FIX] If at start, prefer SEQ1 if present (avoids skipping)
    if state.current_sequence == 0:
        seq1 = next((d for d in seq_dets if CLASS_MAP[d['name']] == 1), None)
        if seq1:
            state.model_sees = seq1['name']
            return 1

    # Pick highest confidence ONLY (Standard rule)
    best = max(seq_dets, key=lambda x: x['confidence'])

    state.model_sees = best['name']
    return CLASS_MAP[best['name']]

def stable_seq(raw_seq):
    """
    Anti-flicker debounce for sequence detection.

    INITIAL detection (current_sequence == 0 → IDLE):
      - Needs only 2 consecutive frames of panel_seq1 before confirming.
      - Fast response — panel just arrived, no flicker risk yet.

    ACTIVE transitions (current_sequence 1→2, 2→3):
      - Needs 4 consecutive frames (was 6).
      - Still prevents false triggers from model flicker, but less delay.
      - At 5fps: 4 frames = 0.8s (was 1.2s with 6 frames).
    """
    val = raw_seq if raw_seq is not None else 0
    state.seq_buf.append(val)

    if val == getattr(state, '_last_raw', -1):
        state._consec_count = getattr(state, '_consec_count', 0) + 1
    else:
        state._consec_count = 1
    state._last_raw = val

    # Threshold depends on context
    if state.current_sequence == 0:
        # IDLE → SEQ1: 3 consecutive frames (fast response)
        threshold = 3
    else:
        # FIX I4: Active transitions (1→2, 2→3): reduced 4→3 consecutive frames.
        # At Pi ~3fps, weak SEQ3 detections (conf~0.39) may miss every 3rd frame;
        # 4-frame requirement means the transition never confirms.
        # 3 frames allows one missed frame while still preventing noise flicker.
        threshold = 3

    if state._consec_count >= threshold:
        state._last_stable = val

    return getattr(state, '_last_stable', 0)


# NOTE: is_wiping() removed — it was dead code (never called) but also
# mutated state.wipe_buf which is owned exclusively by update_wiping().

# OCR trigger functions removed

def capture(seq, frame, metadata=None):
    """
    Save images for a given sequence.
    Robust: guards against None frame and missing panel_rect.
    """
    if frame is None:
        print(f"[CAPTURE] SEQ{seq} skipped — frame is None")
        return None
    if frame.size == 0:
        print(f"[CAPTURE] SEQ{seq} skipped — frame is empty")
        return None

    _SERIAL_PLACEHOLDERS = (None, '', 'UNKNOWN', 'Searching...', 'Reading...')

    # SEQ2/SEQ3: wait up to 2s for Camera-2 OCR to confirm the serial so the
    # saved filename already carries the real serial (avoids SSP-SEQ prefix).
    if seq in (2, 3) and state.serial_number in _SERIAL_PLACEHOLDERS:
        if (getattr(state, 'ocr_started', False)
                and _CAM2_OCR_AVAILABLE
                and camera2_ocr_instance is not None):
            _t0 = time.time()
            while time.time() - _t0 < 2.0:
                _s = camera2_ocr_instance.get_serial_number()
                if _s and _s not in ('UNKNOWN', 'Reading...', 'Searching...', ''):
                    state.serial_number = _s
                    state.ocr_done = True
                    print(f"[CAPTURE] SEQ{seq} serial ready in "
                          f"{time.time()-_t0:.1f}s → {_s}")
                    break
                time.sleep(0.1)
            else:
                print(f"[CAPTURE] SEQ{seq}: serial not ready — "
                      "file saved as SSP-SEQ, renamed at finalization")

    serial = (state.serial_number
              if state.serial_number not in _SERIAL_PLACEHOLDERS
              else 'SSP-SEQ')

    # Use 'is not None' throughout — metadata values can be numpy arrays,
    # and numpy arrays raise ValueError if used with 'or' / bool() checks.
    def _meta(key, fallback_attr):
        v = metadata.get(key) if metadata is not None else None
        return v if v is not None else getattr(state, fallback_attr, None)

    cap_data = {
        'frame':         frame.copy(),
        'panel_rect':    _meta('rect',       'panel_rect'),
        'panel_contour': _meta('contour',    'panel_contour'),
        'serial_det':    _meta('serial_det', 'serial_det'),
        'bbox_xyxy':     _meta('bbox_xyxy',  'panel_bbox_xyxy'),
    }

    saved = capture_sequence_images(cap_data, seq, serial)
    if saved:
        state.sequence_captured[seq] = True
        print(f"[CAPTURE] ✅ SEQ{seq} images saved (serial={serial})")
    else:
        print(f"[CAPTURE] ⚠️ SEQ{seq} capture_sequence_images returned nothing")
    return saved

def complete(seq):
    if seq == 0 or state.completed.get(seq, False): return

    if seq in (2, 3):
        # Use state.sequence_captured[seq] as the authoritative "image saved" flag.
        image_saved = state.sequence_captured.get(seq, False)
        if image_saved:
            print(f"[CAPTURE] SEQ{seq}: ✅ image confirmed saved")
        else:
            # Priority 1: best tracked landscape frame
            bf = (getattr(state, 'seq2_best_frame', None) if seq == 2
                  else getattr(state, 'seq3_best_frame', None))
            if bf is not None:
                print(f"[CAPTURE] SEQ{seq}: rescue via complete() — "
                      f"using landscape tracker best frame "
                      f"(sharp={getattr(state, 'seq2_best_sharp' if seq==2 else 'seq3_best_sharp', 0):.0f})")
                result = capture(seq, bf,
                                 metadata=(getattr(state, 'seq2_best_meta', None) if seq == 2
                                           else getattr(state, 'seq3_best_meta', None)) or {})
                if result:
                    if seq == 2: state.seq2_auto_captured = True
                    else:        state.seq3_auto_captured = True
            else:
                # Priority 2: last hand-free clean frame (landscape only)
                fb = (state.last_clean if state.last_clean is not None
                      else state.orig_frame if state.orig_frame is not None
                      else getattr(state, 'last_clean', None))
                if fb is not None:
                    fh, fw = fb.shape[:2]
                    if fw > fh:   # landscape only — STRICT
                        print(f"[CAPTURE] SEQ{seq}: last-resort rescue with last clean frame ({fw}×{fh})")
                        result = capture(seq, fb)
                        if result:
                            if seq == 2: state.seq2_auto_captured = True
                            else:        state.seq3_auto_captured = True
                    else:
                        print(f"[CAPTURE] SEQ{seq}: ⚠️ last clean frame is portrait ({fw}×{fh}) — skipped")
                else:
                    print(f"[CAPTURE] SEQ{seq}: ⚠️ no frame available — image skipped")
    else:
        # Use state.sequence_captured[seq] to prevent duplicate capture
        image_saved = state.sequence_captured.get(seq, False)
        if image_saved:
            print(f"[CAPTURE] SEQ{seq}: ✅ image confirmed saved, skipping duplicate capture")
        else:
            best_frame = state.seq_capture_data.get(seq)
            if best_frame is None:
                best_frame = getattr(state, 'last_clean', None)
            if best_frame is None:
                best_frame = state.orig if state.orig is not None else None
            if best_frame is None:
                print(f"[CAPTURE] SEQ{seq}: no frame available — capture skipped")
            else:
                capture(seq, best_frame)

    state.completed[seq]         = True
    state.sequence_status[seq]   = 'completed'
    state.seq_end_time[seq]      = time.time()
    
    # Calculate duration if start time exists
    duration = 0
    if state.seq_start_time.get(seq):
        duration = state.seq_end_time[seq] - state.seq_start_time[seq]
        
    if seq < 3:
        state.status_msg = f"SEQ{seq} COMPLETED — FLIP PANEL TO SEQ{seq+1}"
    else:
        state.status_msg = f"SEQ{seq} COMPLETED"
        
    print(f"✅ [TRIGGERED] SEQ{seq} marked as COMPLETED | Duration: {duration:.1f}s")

    # [RULE] When SEQ1 is completed, do NOT immediately disable frame saving.
    # Interval frames are saved at 0s, 1.5s and 3.0s from scan start.
    # Disabling here causes frame-2 and frame-3 to be silently skipped if
    # SEQ1 completes in under 3 seconds (fast operator).
    # camera2_ocr.py self-manages the saving lifecycle — let it finish.
    # Also: only stop OCR scanning if OCR is already confirmed; otherwise the
    # OCR worker thread is mid-run and must not be interrupted.
    if seq == 1:
        if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
            if camera2_ocr_instance.is_done():
                camera2_ocr_instance.stop_scanning()
                print('[CAM2-OCR] SEQ1 complete — OCR already done, scanning stopped')
            else:
                camera2_ocr_instance.stop_yolo_only()
                print('[CAM2-OCR] SEQ1 complete — stopping new Camera-2 detections while background OCR drains captured frames')

    if seq == 3:
        print("\n" + "⭐" * 60)
        print("⭐  SEQ3 COMPLETION TRIGGERED — PANEL REMOVED AFTER WIPING  ⭐")
        print("⭐" * 60 + "\n")
        _finalize_panel()

def alert_missed(missed_seq):
    state.status_msg = f"SEQ{missed_seq} MISSED"
    speak(f"Sequence {missed_seq} missed")

def update_seq(new_seq):
    now = time.time()

    # IDLE — initialize ONLY when SEQ1 is detected (STRICT USER RULE)
    if state.current_sequence == 0:
        if new_seq == 1:
            state.current_sequence = 1
            state.sequence_status[1] = "active"
            state.status_msg = "SEQ1 ACTIVE"
            state.wipe_t0 = None
            state.progress = 0
            state.total_wipe_seconds = 0.0
            state.real_wipe_seconds = 0.0
            state.last_wiping_time = None
            state.cleaning_percentage = 0.0
            state.cleaned_mask = None
            state.wiping_active = False
            state.seq3_wiping_frames = 0
            state.seq3_no_wipe_frames = 0
            state.seq3_completion_stage = 0
            if state.panel_start_time is None:
                state.panel_start_time = now
            if state.seq_start_time.get(new_seq) is None:
                state.seq_start_time[new_seq] = now
            state.current_sequence_panel_name = f"panel_seq{new_seq}"
            print(f"[PIPELINE] Initialized SEQ{new_seq}")
        else:
            # GUARD: block spurious SEQ2/SEQ3 detection before SEQ1 ever started.
            # This is the correct place to prevent orphaned folders — not in
            # ensure_panel_folder() which is called too late.
            print(f"[PIPELINE] ⚠️ Ignored SEQ{new_seq} — SEQ1 never started "
                  f"(current_seq=0). Spurious detection during placement?")
        return

    # Same sequence — no change
    if new_seq == state.current_sequence:
        return


    # Higher sequence detected → operator moved to next area (only advance one step)
    if new_seq == state.current_sequence + 1:
        # ── STRICT SEQUENCE ENFORCEMENT ───────────────────────────
        # Reject transition if the current sequence has not been fully completed
        if not state.completed.get(state.current_sequence, False):
            if getattr(state, '_last_ignore_log', 0) < time.time() - 2.0:
                print(f"[PIPELINE] ⚠️ Ignored SEQ{new_seq} class detection — "
                      f"SEQ{state.current_sequence} wiping is not yet 100% completed.")
                state._last_ignore_log = time.time()
            return

        prev = state.current_sequence

        # ── SEQ3 Transition Guard ─────────────────────────────────
        # [FIX] Do NOT force-complete SEQ2 here — the 4-condition gate in
        # update_wiping() handles completion with proper timing (3s + 30% + 5 frames).
        # This block ONLY ensures the image is saved if it wasn't captured yet.
        if new_seq == 3 and prev == 2:
            if not getattr(state, 'seq2_auto_captured', False):
                _bf2 = getattr(state, 'seq2_best_frame', None)
                if _bf2 is not None:
                    print('[SEQ2] ⚡ Safety: force-capturing SEQ2 image at SEQ3 detection')
                    capture(2, _bf2, metadata=getattr(state, 'seq2_best_meta', {}))
                    state.seq2_auto_captured = True
                    if not hasattr(state, 'seq_capture_status'): state.seq_capture_status = {}
                    state.seq_capture_status[2] = "CAPTURED_AT_SEQ3_TRANSITION"

        state.current_sequence = new_seq
        state.sequence_status[new_seq] = "active"

        # ── Landscape voice + screen alert for SEQ2 and SEQ3 ────────
        # NOTE: For SEQ2, do NOT set landscape_alert here — the operator has
        # just flipped the panel and wiping hasn't started yet.
        # The landscape/image-capture prompt fires from update_wiping() at
        # 25–35% wipe progress (see _seq2_prompted gate below).
        if new_seq == 2:
            # Do NOT set landscape_alert = "seq2" here — premature banner.
            # _announce_landscape(2) is also called from update_wiping() at 25-35%.
            if not state.completed.get(2, False):
                state.status_msg = "SEQ2 ACTIVE — start wiping"

        # ── SAVE final wipe % AND wiping frames for the PREVIOUS sequence ──
        # BEFORE progress/counters reset.  These saved values are the
        # authoritative records used by the SEQ1/SEQ2/SEQ3 completion gates.
        if prev == 1:
            state.seq1_final_wipe_pct = state.progress
            print(f'[SEQ1] 💾 Final wipe pct saved = {state.seq1_final_wipe_pct}% '
                  f'before progress reset for SEQ{new_seq}')
        elif prev == 2:
            state.seq2_final_wipe_pct = state.progress
            state.seq2_final_wiping_frames = state.seq2_wiping_frames
            print(f'[SEQ2] 💾 Final wipe pct saved = {state.seq2_final_wipe_pct}% '
                  f'wiping_frames saved = {state.seq2_final_wiping_frames} '
                  f'before progress reset for SEQ{new_seq}')
            # Reset SEQ1-gate stability counter (only used while in SEQ2)
            state.seq2_stable_for_seq1 = 0
            # [SEQ2 LEAD TIME] Freeze seq_end_time[2] at the moment SEQ3 activates
            # so the UI shows a fixed (correct) SEQ2 duration from this point on.
            if state.seq_end_time.get(2) is None and state.seq_start_time.get(2):
                state.seq_end_time[2] = now
                print(f'[SEQ2] ⏱️ Lead time frozen at SEQ2→SEQ3 transition: '
                      f'{round(now - state.seq_start_time[2], 1)}s')

            # ── FORCE CAPTURE SEQ2 image if not already done ──────────────
            # When SEQ3 appears suddenly (fast operator, strong model detection),
            # seq2_best_frame may already be populated but capture() not yet called.
            # Capture it NOW before the transition resets landscape trackers.
            if not state.seq2_auto_captured:
                _bf2 = state.seq2_best_frame
                if _bf2 is not None:
                    print('[SEQ2] ⚡ Force-capturing SEQ2 at SEQ2→SEQ3 transition '
                          f'(best_sharp={state.seq2_best_sharp:.0f})')
                    capture(2, _bf2, metadata=state.seq2_best_meta or {})
                    state.seq2_auto_captured = True
                    # [SEQ2 GATE] Start post-capture timer right now
                    if state.seq2_capture_time is None:
                        state.seq2_capture_time = now
                        state.seq2_post_capture_wipe_pct = state.seq2_final_wipe_pct
                        print(f'[SEQ2] 📸 Force-capture at SEQ2→SEQ3: '
                              f'post-capture gate started (wipe={state.seq2_final_wipe_pct}%)')
                else:
                    # No landscape frame yet — use last clean hand-free frame
                    _fb = (state.last_clean if state.last_clean is not None
                           else state.orig)
                    if _fb is not None:
                        fh, fw = _fb.shape[:2]
                        if fw > fh:   # landscape only
                            print('[SEQ2] ⚡ Force-capturing SEQ2 using last clean frame')
                            capture(2, _fb)
                            state.seq2_auto_captured = True
                            if state.seq2_capture_time is None:
                                state.seq2_capture_time = now
                                state.seq2_post_capture_wipe_pct = state.seq2_final_wipe_pct
                        else:
                            print('[SEQ2] ⚠️ No landscape frame available for SEQ2 force-capture')
                    else:
                        print('[SEQ2] ⚠️ No frame available for SEQ2 force-capture')
        elif prev == 3:
            state.seq3_final_wipe_pct = state.progress

        state.wipe_t0 = None
        state.progress = 0
        state.total_wipe_seconds = 0.0
        state.real_wipe_seconds = 0.0
        state.last_wiping_time = None
        state.cleaning_percentage = 0.0
        state.cleaned_mask = None
        state.wiping_active = False
        
        # Reset counters for the new sequence
        state.seq1_wiping_frames = 0
        state.seq2_wiping_frames = 0
        state.seq3_wiping_frames = 0
        state.seq3_no_wipe_frames = 0
        state.seq3_completion_stage = 0
        state.seq3_wiping_started = False
        state.seq3_consec_wipe_frames = 0   # strict consecutive counter — reset on seq change
        
        if new_seq == 3:
            state.status_msg = "SEQ3 ACTIVE"
        elif new_seq == 2:
            state.status_msg = "SEQ2 ACTIVE"
        else:
            state.status_msg = f"SEQ{new_seq} ACTIVE"
        if state.seq_start_time.get(new_seq) is None:
            state.seq_start_time[new_seq] = now
        state.current_sequence_panel_name = f"panel_seq{new_seq}"
        state.reset_sequence_timers()
        print(f"[PIPELINE] Moved to SEQ{new_seq}")

        # [RULE] When SEQ2 starts the panel has been flipped.
        # ONLY stop Camera-2 OCR if it has already confirmed the serial.
        # If OCR is still collecting frames / running OCR, keep it alive —
        # stopping early causes interval frames 2 & 3 and capture slots to be lost.
        if new_seq == 2:
            if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
                if camera2_ocr_instance.is_done():
                    camera2_ocr_instance.stop_scanning()
                    print('[CAM2-OCR] ✅ OCR already confirmed — scanning stopped at SEQ2 start')
                else:
                    print('[CAM2-OCR] ⚠️  SEQ2 started but OCR NOT done — '
                          'keeping OCR alive to finish 3-slot capture + interval frames')

def update_wiping(hand_present, motion):
    """
    Unified wipe tracking with new sequence flow:

    SEQ1: class detected → hand-free 2s delay → capture photo (no hand)
          → serial cam activates → operator wipes → progress bar fills → done
    SEQ2: class detected → capture photo (first stable landscape, no hand)
          → operator wipes → progress bar fills over PROGRESS_BAR_DURATION_SECS → done
    SEQ3: same as SEQ2

    Progress bar: starts filling AFTER photo is captured, driven by wiping time.
    Fills to 100% over PROGRESS_BAR_DURATION_SECS seconds of cumulative wiping.
    """
    # Wiping = hand present AND (motion OR hand overlaps panel)
    wiping_now = hand_present and (motion or (
        state.hand_bbox is not None and state.panel_mask is not None and
        state.panel_mask.shape[0] > 0))

    state.wipe_buf.append(1 if wiping_now else 0)
    wiping_confirmed = sum(state.wipe_buf) >= 2  # 2 out of 3 frames

    now = time.time()
    cur = state.current_sequence

    # ── Hand-on-panel check ─────────────────────────────────────
    _hand_on_panel = (
        state.hand_bbox is not None
        and state.panel_mask is not None
        and state.panel_mask.shape[0] > 0
    )

    # ── ALL ZONES: hand-free delay before capture ────────────────────
    is_captured = getattr(state, f'seq{cur}_auto_captured', False)
    
    if not is_captured:
        _multi = getattr(state, 'multiple_panels_visible', False)
        if not hand_present and not _multi:
            # Hand is absent AND panel is flat — start or continue the delay timer
            if state.seq_hand_free_since[cur] is None:
                state.seq_hand_free_since[cur] = now
                print(f'[SEQ{cur}-DELAY] Condition met — starting {CAPTURE_DELAY_SECS}s capture delay timer')
            state.seq_phase[cur] = 'hand_clear'
            elapsed_free = now - state.seq_hand_free_since[cur]
            # Update status with countdown
            remaining = max(0.0, CAPTURE_DELAY_SECS - elapsed_free)
            if remaining > 0:
                state.status_msg = f"SEQ{cur} — Hold still... {remaining:.1f}s"
            else:
                state.status_msg = f'SEQ{cur} — Capturing...'
        else:
            # Hand is present or panel is tilted — reset the delay timer
            if state.seq_hand_free_since[cur] is not None:
                print(f'[SEQ{cur}-DELAY] Condition broken (hand={hand_present}, multi={_multi}) — resetting capture delay timer')
            state.seq_hand_free_since[cur] = None
            state.seq_phase[cur] = 'waiting_class'
            if _multi:
                state.status_msg = f'SEQ{cur} — Lay panel flat (multiple zones seen)'
            else:
                state.status_msg = f'SEQ{cur} — Remove hand for capture'

    # ── SEQ2/SEQ3: progress bar driven by post-capture wipe time ─
    if cur in (2, 3):
        if not is_captured:
            pass # handled above
        elif not state.completed.get(cur, False):
            # Photo taken — accumulate wipe time only when hand is on panel
            if wiping_confirmed and _hand_on_panel:
                last_w = state.post_capture_last_wipe.get(cur)
                if last_w is not None:
                    _dt = now - last_w
                    if _dt < 2.0:  # guard against gaps/pauses
                        state.post_capture_wipe_secs[cur] = (
                            state.post_capture_wipe_secs.get(cur, 0.0) + _dt)
                state.post_capture_last_wipe[cur] = now
                state.seq_phase[cur] = 'wiping'
                state.wiping_active = True
                state.sequence_status[cur] = 'wiping'
                if state.wipe_t0 is None:
                    state.wipe_t0 = now
            else:
                # Not wiping right now
                state.post_capture_last_wipe[cur] = None
                if state.seq_phase.get(cur) == 'wiping':
                    state.wiping_active = False

            # Progress = accumulated wipe time / total bar duration
            wipe_secs = state.post_capture_wipe_secs.get(cur, 0.0)
            state.progress = min(100, int((wipe_secs / PROGRESS_BAR_DURATION_SECS) * 100))
            state.cleaning_percentage = float(state.progress)

            if getattr(state, '_last_debug_print', 0) < now - 1.0:
                print(f'[DEBUG] SEQ{cur} Post-capture wipe: {state.progress}% '
                      f'({wipe_secs:.1f}s / {PROGRESS_BAR_DURATION_SECS}s)')
                state._last_debug_print = now

            pct_rem = 100 - state.progress
            state.status_msg = (
                f'SEQ{cur}: Photo taken — wipe panel ({state.progress}%)'
                if state.progress < 100
                else f'SEQ{cur}: Wipe complete ✅'
            )

            # COMPLETION: photo captured + progress bar reached 100%
            if state.progress >= 100:
                print(f'[SEQ{cur}] ✅ 100% wipe complete — marking COMPLETED')
                setattr(state, f'seq{cur}_final_wipe_pct', state.progress)
                state.seq_phase[cur] = 'done'
                complete(cur)
                return

            # Track post-capture wipe frames for SEQ2 completion gate
            if _hand_on_panel:
                setattr(state, f'_seq{cur}_post_cap_wipe_frames',
                        getattr(state, f'_seq{cur}_post_cap_wipe_frames', 0) + 1)

    # ── SEQ1 post-capture wipe tracking (legacy total_wipe_seconds) ──
    if wiping_confirmed:
        state.wiping_active = True
        if state.wipe_t0 is None:
            state.wipe_t0 = now
            if cur == 1:
                state.status_msg = 'SEQ1 WIPING'
                state.sequence_status[cur] = 'wiping'

        if getattr(state, 'last_wiping_time', None) is not None:
            elapsed = now - state.last_wiping_time
            if elapsed < 2.0 and cur == 1:
                state.total_wipe_seconds = min(
                    PROGRESS_BAR_DURATION_SECS,
                    getattr(state, 'total_wipe_seconds', 0.0) + elapsed)
        state.last_wiping_time = now

        if cur == 1:
            # SEQ1 progress bar driven by wipe seconds (only after capture)
            is_cap1 = getattr(state, 'seq1_auto_captured', False)
            if is_cap1:
                state.progress = min(100, int(
                    (state.total_wipe_seconds / PROGRESS_BAR_DURATION_SECS) * 100))
                state.cleaning_percentage = float(state.progress)
            else:
                state.progress = 0
                state.cleaning_percentage = 0.0

            if getattr(state, '_last_debug_print', 0) < now - 1.0:
                print(f'[DEBUG] SEQ1 Wipe: {state.progress}% '
                      f'({state.total_wipe_seconds:.1f}s / {PROGRESS_BAR_DURATION_SECS:.1f}s)')
                state._last_debug_print = now

            state.seq1_wiping_frames += 1
            # Trigger Camera-2 OCR when SEQ1 wiping starts (serial face visible)
            if state.seq1_wiping_frames >= 3 and not state.ocr_started:
                if state.seq1_snapshot_data is None:
                    print('[CAM2-OCR] ⚠️  SEQ1 snapshot not confirmed yet — OCR trigger BLOCKED')
                    if state.seq1_wiping_frames >= 6:
                        folder = ensure_panel_folder()
                        if folder and camera2_ocr_instance is not None:
                            camera2_ocr_instance.set_panel_folder(folder)
                            camera2_ocr_instance.start_ocr()
                            state.ocr_started = True
                            print(f'[CAM2-OCR] ⚠️  Fallback OCR triggered at frame {state.seq1_wiping_frames}')
                            state.ocr_done = False
                else:
                    folder = ensure_panel_folder()
                    if folder and camera2_ocr_instance is not None:
                        camera2_ocr_instance.set_panel_folder(folder)
                        camera2_ocr_instance.start_ocr()
                        state.ocr_started = True
                        print(f'[CAM2-OCR] ✅ OCR triggered by SEQ1 wiping (frame {state.seq1_wiping_frames})')
                        state.ocr_done = False

    # ── SEQ1 status message update (happens whether wiping or not) ──
    if cur == 1:
        is_captured = getattr(state, 'seq1_auto_captured', False)
        # Check if delay timer logic already set the status
        _delay_handled = (state.seq_hand_free_since.get(1) is not None)
        
        if not is_captured and not _delay_handled:
            state.status_msg = 'SEQ1 — Waiting for photo capture...'
        elif is_captured and not state.completed.get(1, False):
            if state.progress < 100:
                state.status_msg = f'SEQ1: Photo taken — wipe panel ({state.progress}%)'
            else:
                state.status_msg = 'SEQ1: Wipe complete — flip to SEQ2'

            if state.progress >= 100:
                print(f'[SEQ1] ✅ Photo captured + wiped ({state.progress}%) — marking COMPLETED')
                state.seq1_final_wipe_pct = state.progress
                state.seq_phase[1] = 'done'
                complete(1)

    # ── No-wipe: reset wiping_active ──────────────────────────
    if not wiping_confirmed:
        state.wiping_active = False

    if cur == 3:
            # [FIX] Count SEQ3 wipe frames when hand is confirmed ON the panel.
            # The old gate (current_panel_name == "panel_seq3") was too strict:
            # during SEQ2→SEQ3 transition both panel classes appear together and
            # the model may still output panel_seq2, making the counter stuck at 0.
            # We still require the panel to be visible (panel_mask present) so
            # false positives from gloves on an empty table are blocked.
            # seq3_wiping_started is set only when panel_seq3 is the confirmed class
            # (unchanged) — this guards the SEQ3 wipe-% progress gate.
            _hand_on_panel_s3 = (
                state.hand_bbox is not None
                and state.panel_mask is not None
                and state.panel_mask.shape[0] > 0
            )
            if state.current_panel_name == "panel_seq3":
                state.seq3_wiping_started = True   # only set when SEQ3 class confirmed

            if _hand_on_panel_s3:
                state.seq3_wiping_frames += 1
                # Strict consecutive counter — resets to 0 on any non-wipe frame.
                state.seq3_consec_wipe_frames = \
                    getattr(state, 'seq3_consec_wipe_frames', 0) + 1
                # [SEQ2 COMPLETION GATE] Count SEQ3 wipe frames after SEQ2 image captured.
                # Once this reaches 5, the SEQ2 4-condition gate can fire.
                if state.seq2_auto_captured and not state.completed.get(2, False):
                    state.seq2_post_cap_seq3_frames = \
                        getattr(state, 'seq2_post_cap_seq3_frames', 0) + 1
                    print(f"[SEQ2-GATE] SEQ3 wipe frame #{state.seq2_post_cap_seq3_frames} "
                          f"counted toward SEQ2 completion gate "
                          f"(need 5, have {state.seq2_post_cap_seq3_frames})")
                print(f"[SEQ3] Hand ON panel — wipe frame #{state.seq3_wiping_frames} "
                      f"consec={state.seq3_consec_wipe_frames} "
                      f"(panel_name={state.current_panel_name})")
            state.seq3_no_wipe_frames = 0

            # NOTE: SEQ3 capture is handled EXCLUSIVELY by the landscape tracker.
            # Never store last_clean/state.orig here — those may be SEQ1/empty frames.

            # [RULE] SEQ3 completion handled when panel disappears in process_frame
            # but we need to reach 90% first (raised from 85%).
            if state.progress >= 90:
                print(f"[SEQ3] 90% reached — waiting for panel to fully disappear")

    else:
        # DO NOT reset state.last_wiping_time = None here, otherwise flickering drops accumulation
        state.wiping_active    = False
        if hand_present:
            state.hand_gone_since = None
        elif state.hand_gone_since is None:
            state.hand_gone_since = now

        # SEQ3: reset wipe counter only if hand gone for long time
        if cur == 3:
            state.seq3_no_wipe_frames = getattr(state, 'seq3_no_wipe_frames', 0) + 1
            # Reset strict consecutive counter on ANY non-wipe frame
            state.seq3_consec_wipe_frames = 0
            if state.seq3_wiping_frames < 5 and state.seq3_no_wipe_frames >= 3:
                state.seq3_wiping_frames = 0
                state.seq3_no_wipe_frames = 0
                print(f"[SEQ3] Wiping interrupted — reset")

    # ────────────────────────────────────────────────────────────
    # SEQ2 COMPLETION GATE — runs OUTSIDE wiping_confirmed block so it fires
    # every frame while in SEQ3, even when operator briefly lifts hand.
    # The 4 conditions must ALL be true before complete(2) is called.
    # ────────────────────────────────────────────────────────────
    if cur == 3 and not state.completed.get(2, False):
        # Step 1: Make sure seq2_capture_time is set (catches landscape-tracker captures)
        if state.seq2_auto_captured and state.seq2_capture_time is None:
            state.seq2_capture_time = time.time()
            state.seq2_post_capture_wipe_pct = 0
            print("[SEQ2-GATE] 📸 capture_time set (landscape tracker capture detected in SEQ3 block)")

        # Step 2: Evaluate conditions
        if state.seq2_auto_captured and state.seq2_capture_time is not None:
            _secs  = time.time() - state.seq2_capture_time
            _s3_ok = getattr(state, 'seq2_post_cap_seq3_frames', 0) >= 5
            _w_ok  = getattr(state, '_seq2_post_cap_wipe_frames', 0) >= 3
            _t_ok  = _secs >= 15.0  # [FIX] Require full 15s post-capture wait
            print(f"[SEQ2-GATE] ► secs={_secs:.1f}s "
                  f"seq3_frames={getattr(state,'seq2_post_cap_seq3_frames',0)} "
                  f"| time={_t_ok} wipe={_w_ok} seq3={_s3_ok}")
            if _t_ok and _w_ok and _s3_ok:
                print("[SEQ2] ✅ Time, Wipe and SEQ3 conditions met — marking SEQ2 COMPLETED")
                complete(2)

    # [RULE] Sequence completion logic has been moved to process_frame 
    # to support activity-based transitions (e.g. SEQ3 wiping confirms SEQ2 done).


def reset_panel():
    """Full system reset — called after PDF generation or when a new panel_seq1 arrives.
    Increments panel_reset_id ONCE so the frontend status poller fires resetAllProgressBars
    exactly once per panel cycle.
    """
    # Call reset_for_new_panel() which covers ALL fields including
    # _capture_slots, _slots_saved, intervals_saved that the old
    # piecemeal reset was missing — old scan data was bleeding into
    # the next panel and triggering stale OCR worker signals.
    global camera2_ocr_instance
    if camera2_ocr_instance is not None:
        try:
            camera2_ocr_instance.reset_for_new_panel()
        except Exception as e:
            print(f"[CAM2] reset_for_new_panel error: {e}")

    state.reset_for_new_panel()          # handles all per-panel state + increments panel_reset_id
    state.current_sequence = 0
    state.completed        = {1: False, 2: False, 3: False}
    state.all_sequences_done = False
    state.wipe_t0          = None
    state.progress         = 0
    state.ocr_done         = False
    state.serial           = None
    state.serial_number    = "Searching..."
    state.partial_serial   = None     # clear so UI hint resets cleanly
    state.seq_buf.clear()
    state.wipe_buf.clear()
    state._last_raw        = -1
    state._last_stable     = 0
    state._consec_count    = 0
    state.status_msg       = "READY (NEW PANEL)"
    
    # ── Camera-2 OCR reset (status message only — fields cleared above) ──
    if camera2_ocr_instance is not None:
        try:
            camera2_ocr_instance.status = "Waiting for new panel..."
        except Exception: pass
        
    print(f"[RESET] Panel reset complete — reset_id={state.panel_reset_id}")
    # Clear held annotation boxes so previous panel's bboxes don't bleed through
    with _overlay_lock:
        _held_panel.update( {'bbox': None, 'label': '', 'color': (200,200,200), 'ttl': 0})
        _held_hand.update(  {'bbox': None, 'ttl': 0})
        _held_serial.update({'bbox': None, 'ttl': 0})

# ─────────────────────────────────────────────────────────────────────────────
# STABLE STATE-DRIVEN OVERLAY RENDERING
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Never draw raw model predictions directly.
# Instead, maintain "held" boxes that survive for PANEL_HOLD_FRAMES after
# the last confirmed detection.  The panel box is driven by the state machine
# (confirmed sequence), not the noisy per-frame model output.
# This eliminates ALL bbox flicker regardless of backend (CPU or Hailo).
# ─────────────────────────────────────────────────────────────────────────────

# ── How many encoder frames to keep showing a box after the last detection ──
PANEL_HOLD_FRAMES = 15   # ~1 s at 15 FPS stream
HAND_HOLD_FRAMES  = 6    # ~0.4 s — hand boxes fade faster

# ── Per-class BGR colours ─────────────────────────────────────────────────────
_CLASS_COLORS = {
    'panel_seq1':           (0,   0, 230),    # Red
    'panel_seq2':           (0, 140, 255),    # Orange
    'panel_seq3':           (0, 210,  20),    # Green
    'hand':                 (0, 215, 255),    # Yellow
    'serial_number':        (255,  80,  80),  # Blue
}

# ── Human-readable display names ─────────────────────────────────────────────
_DISPLAY_NAME = {
    'panel_seq1':           'Panel  SEQ-1',
    'panel_seq2':           'Panel  SEQ-2',
    'panel_seq3':           'Panel  SEQ-3',
    'hand':                 'Hand',
    'serial_number':        'Serial No.',
    'serial_number_region': 'Serial No.',
}

# ── Rendering constants ───────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE = 0.70
_FONT_THICK = 2
_BOX_THICK  = 3   # slightly thicker for crisp look

# ── Held-box state (updated by infer thread, read by encoder thread) ─────────
# Each entry: {'bbox': (x1,y1,x2,y2), 'label': str, 'color': BGR, 'ttl': int}
_overlay_lock = threading.Lock()
_held_all_boxes = []


def _det_color(name):
    return _CLASS_COLORS.get(name, (200, 200, 200))


def _display(name):
    return _DISPLAY_NAME.get(name, name.replace('_', ' ').title())


def _draw_label(frame, text, x, y, color):
    """
    Near-black pill background  +  coloured border  +  white text.
    Positioned above the top-left corner of the bounding box.
    """
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICK)
    pad  = 5
    bx1  = x
    by1  = max(0, y - th - pad * 2)
    bx2  = x + tw + pad * 2
    by2  = y
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(frame, text, (bx1 + pad + 1, by2 - pad + 1),
                _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad),
                _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICK, cv2.LINE_AA)


def update_held_boxes(detections, conf_seq):
    """
    Called by the INFER thread after process_frame().
    Updates the held-box state from the freshly confirmed detections.
    (Now showing ALL detected classes regardless of sequence)
    """
    global _held_all_boxes
    
    new_boxes = []
    for d in detections:
        conf = d.get('confidence', 0.0)
        name = d.get('name', '')
        
        # Different confidence thresholds based on class type
        threshold = 0.25 if name == 'hand' else 0.40
        
        if conf >= threshold:
            bbox = d.get('bbox')
            color = _CLASS_COLORS.get(name, (200, 200, 200))
            if name.startswith('panel_seq'):
                label = f"{_DISPLAY_NAME.get(name, name)} {int(conf*100)}%"
            elif name == 'hand':
                label = f"Hand {int(conf*100)}%"
            elif name in ('serial_number', 'serial_number_region'):
                label = f"Serial No. {int(conf*100)}%"
            else:
                label = f"{name} {int(conf*100)}%"
                
            new_boxes.append({'bbox': bbox, 'label': label, 'color': color})
            
    with _overlay_lock:
        _held_all_boxes = new_boxes


def draw_overlay(frame, seq, status, progress):
    try:
        return _draw_overlay_internal(frame, seq, status, progress)
    except Exception as e:
        import traceback, cv2
        traceback.print_exc()
        try:
            cv2.putText(frame, f"UI CRASH: {e}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        except: pass
        return frame

def _draw_overlay_internal(frame, seq, status, progress):
    """
    State-driven overlay — draws held boxes, not raw model predictions.
    Boxes persist for PANEL_HOLD_FRAMES / HAND_HOLD_FRAMES frames after
    the last confirmed detection, so there is ZERO flicker.
    """
    with _overlay_lock:
        boxes_to_draw = list(_held_all_boxes)

    for box_info in boxes_to_draw:
        try:
            bbox = box_info['bbox']
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = box_info['color']
                label = box_info['label']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, _BOX_THICK)
                _draw_label(frame, label, x1, y1, color)
        except Exception:
            pass


    return frame



# ─────────────────────────────────────────────────────────────
# MAIN FRAME PROCESSING
# ─────────────────────────────────────────────────────────────

def process_frame(frame, detections):
    try:
        return _process_frame_internal(frame, detections)
    except Exception as e:
        import traceback
        traceback.print_exc()
        state.status_msg = f"CRASH: {e}"
        return None

def _process_frame_internal(frame, detections):
    """Processes a single frame for the state machine."""
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # ── 1. Filter detections ─────────────────────────────────────
    # FIX I1: TRACK A and TRACK B use DIFFERENT confidence thresholds.
    #
    # Production log showed panel_seq3 at conf=0.39 during SEQ2 wiping
    # (misclassification: model sees panel_seq1 on SEQ2 face at 0.94 conf).
    # With a shared 0.50 threshold, panel_seq3 was invisible to TRACK A so
    # the state machine never advanced to SEQ3 and no PDF was generated.
    #
    # TRACK A (state machine):  0.35 when panel active (allows weak
    #   cross-face transitions like panel_seq3 at 0.39 to advance state);
    #   0.50 when IDLE (prevents ghost badges on empty table).
    # TRACK B (captures only):  always 0.50 (ensures sharp, correct captures).
    _idle = (state.current_sequence == 0)
    _thresh_A = 0.60 if _idle else 0.35   # IDLE needs 0.60 (harder gate on empty table)
    _thresh_B = 0.50                       # captures

    def _valid_panel(d, min_area=0.00):
        bx1, by1, bx2, by2 = d['bbox']
        area = max(0, bx2 - bx1) * max(0, by2 - by1)
        return area > 0

    seq_cands_A = [d for d in detections if d['name'].startswith('panel_seq')
                   and d['confidence'] >= _thresh_A and _valid_panel(d)]
    seq_cands_B = [d for d in detections if d['name'].startswith('panel_seq')
                   and d['confidence'] >= _thresh_B and _valid_panel(d)]
    hand_candidates = [d for d in detections if d['name'] == 'hand'
                       and d['confidence'] >= 0.5]
    serial_candidates = [d for d in detections
                         if d['name'] in ('serial_number', 'serial_number_region')
                         and d['confidence'] >= 0.3]

    # ── 2. Select BEST panel detection — TWO-TRACK logic ───────────
    #
    # TRACK A (raw_seq_id / best_any_panel):
    #   Low threshold — picks highest-confidence panel for state transitions.
    #   Allows weak SEQ3 detections (0.35+) to advance the state machine.
    #
    # TRACK B (panel_det):
    #   High threshold, class-locked — only used for capture decisions.
    #   Prevents wrong-face or blurry captures.

    # TRACK A — best detection of ANY panel class (drives state machine)
    best_any_panel = (max(seq_cands_A, key=lambda x: x['confidence'])
                      if seq_cands_A else None)
    raw_seq_id = 0
    if best_any_panel:
        try:
            raw_seq_id = int(best_any_panel['name'].replace('panel_seq', ''))
        except Exception:
            raw_seq_id = 0

    # ── GATE: SEQ1 from IDLE — panel size only ───────────────────────────────
    # We now filter all panels globally to be >= 15% of the frame.
    # For IDLE -> SEQ1 transitions, we enforce a stricter 25% area requirement
    # to completely eliminate empty table false positives.
    if raw_seq_id == 1 and state.current_sequence == 0:
        panel_big_enough = False
        if best_any_panel:
            bx1, by1, bx2, by2 = best_any_panel['bbox']
            panel_px = max(0, bx2 - bx1) * max(0, by2 - by1)
            panel_big_enough = (panel_px >= 0.25 * h * w)

        if not panel_big_enough:
            # Panel bbox too small — likely a false positive
            raw_seq_id = 0
            state.seq1_detection_count = 0

    # TRACK B — class-locked detection (drives captures only)
    panel_det = None
    if seq_cands_B:
        expected_seq_id = state.current_sequence if state.current_sequence > 0 else 1
        expected_class  = f"panel_seq{expected_seq_id}"
        exact_matches   = [d for d in seq_cands_B if d['name'] == expected_class]
        if exact_matches:
            panel_det = max(exact_matches, key=lambda x: x['confidence'])
        else:
            # Log the mismatch (info only — state machine will handle transition)
            if best_any_panel:
                print(f"[FILTER] Capture locked to {expected_class} — "
                      f"seen '{best_any_panel['name']}' "
                      f"(conf={best_any_panel['confidence']:.2f}) [state machine ok]")


    # Best hand and serial
    hand_det   = max(hand_candidates,   key=lambda x: x['confidence']) \
                 if hand_candidates else None
    serial_det = max(serial_candidates, key=lambda x: x['confidence']) \
                 if serial_candidates else None

    # ── 4. Update shared state ────────────────────────────────────
    state.hand_bbox  = hand_det['bbox']  if hand_det   else None
    state.serial_det = serial_det        if serial_det else None
    hand_present     = state.hand_bbox is not None
    # NEW: detect if panel is tilted/flipping (showing multiple seqs)
    state.multiple_panels_visible = len(set(d['name'] for d in seq_cands_A)) > 1
    state.orig       = frame.copy()
    if not hand_present:
        state.last_clean = frame.copy()

    if serial_det:
        s_conf = serial_det['confidence']
        # FIRST
        if getattr(state, 'serial_first_trace', None) is None:
            state.serial_first_trace = (state.frame_count, s_conf)
            log.serial_trace("FIRST", state.frame_count, s_conf)
        
        # BEST
        best = getattr(state, 'serial_best_trace', (0, 0.0))
        if s_conf > best[1]:
            state.serial_best_trace = (state.frame_count, s_conf)
            log.serial_trace("BEST", state.frame_count, s_conf)
            
        # LAST
        state.serial_last_trace = (state.frame_count, s_conf)

    # [RULE] Trigger snapshot when panel_seq1 visible, no hand blocking.
    # NEW: Also enforce CAPTURE_DELAY_SECS (hand must be absent that long).
    # has_serial removed — best.hef does not detect serial_number class.
    # Protection against empty table: _thresh_A=0.60 + panel size gate above.
    is_panel_seq1 = (raw_seq_id == 1)

    if is_panel_seq1 and not hand_present and not getattr(state, 'multiple_panels_visible', False) \
            and state.current_sequence in (0, 1):
        state.seq1_detection_count = getattr(state, 'seq1_detection_count', 0) + 1

        # ── NEW: hand-free delay gate ────────────────────────────────
        _delay_satisfied = False
        _hand_free_elapsed = 0.0
        if state.seq_hand_free_since[1] is not None:
            _hand_free_elapsed = time.time() - state.seq_hand_free_since[1]
            _delay_satisfied   = (_hand_free_elapsed >= CAPTURE_DELAY_SECS)

        if serial_det is not None:
            state.seq1_serial_stable_count = getattr(state, 'seq1_serial_stable_count', 0) + 1
        else:
            state.seq1_serial_stable_count = 0

        if state.seq1_detection_count == 1:
            state.status_msg = "SEQ1 Panel Detected — stabilising..."
            print(f"[SEQ1-DET] Frame {state.frame_count}: Panel detected (conf={raw_seq_id})")

        should_capture = False
        # NORMAL: consecutive clean frames WITH serial_no AND delay satisfied
        if _delay_satisfied and getattr(state, 'seq1_serial_stable_count', 0) >= SEQ1_NORMAL_FRAMES:
            should_capture = True
            print(f"[SEQ1-DET] ✅ Frame {state.frame_count}: Panel + Serial stable for {SEQ1_NORMAL_FRAMES} frames "
                  f"+ {_hand_free_elapsed:.1f}s hand-free — capturing")
            print(f"[SEQ1-DET]    current_sequence: {state.current_sequence}")
        # FALLBACK: consecutive clean frames WITHOUT serial_no AND delay satisfied
        elif _delay_satisfied and state.seq1_detection_count >= SEQ1_FALLBACK_FRAMES:
            should_capture = True
            print(f"[SEQ1-DET] ⚠️ Frame {state.frame_count}: {SEQ1_FALLBACK_FRAMES} frames without serial_no "
                  f"(hand free {_hand_free_elapsed:.1f}s) — triggering fallback")
        elif not _delay_satisfied:
            remaining = CAPTURE_DELAY_SECS - _hand_free_elapsed
            state.status_msg = f"SEQ1 — Hold still... {remaining:.1f}s"

        if should_capture and state.seq1_snapshot_data is None:
            frame_sharp = cv2.Laplacian(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

            panel_is_stable = True
            if frame_sharp < 12:  # FIX D2: 25->12 for Pi H.264 RTSP
                panel_is_stable = False
                print(f"[SEQ1] Frame too blurry (sharpness={frame_sharp:.0f} < 12) - waiting")

            if panel_is_stable:
                if panel_det is not None:
                    bx1, by1, bx2, by2 = panel_det['bbox']
                    current_rect = (bx1, by1, bx2 - bx1, by2 - by1)
                    # [RULE] Panel must be clearly horizontal (Landscape)
                    if (bx2 - bx1) < ((by2 - by1) * 1.05):  # FIX D2: 1.15->1.05 for Pi bbox rounding
                        print(f"[SEQ1] Panel is portrait - capturing anyway as requested")
                    else:
                        pass
                else:
                    # Fallback if only serial_number was seen: use full frame as panel_rect
                    current_rect = (0, 0, frame.shape[1], frame.shape[0])

            # Only commit snapshot when ALL stability checks pass
            if panel_is_stable:
                state.seq1_snapshot_data = {
                    'frame':      frame.copy(),
                    'rect':       current_rect,
                    'contour':    getattr(state, 'panel_contour', None),
                    'serial_det': serial_det,
                }
                state.seq1_first_clean_frame = state.seq1_snapshot_data['frame']
                # [NEW] Trigger folder creation immediately
                ensure_panel_folder()
                if not hasattr(state, 'seq_capture_status'): state.seq_capture_status = {}
                if serial_det is None:
                    print(f"[SEQ1] ✅ Snapshot taken WITHOUT serial_number (Fallback) at frame {state.seq1_detection_count} — sharpness={frame_sharp:.0f}")
                    state.seq1_snapshot_data['fallback'] = True
                    state.seq_capture_status[1] = "FALLBACK_NO_SERIAL"
                    log.capture_event("SEQ1", "FALLBACK_NO_SERIAL")
                else:
                    print(f"[SEQ1] ✅ Snapshot taken WITH serial_number at frame {state.seq1_detection_count} — sharpness={frame_sharp:.0f}")
                    state.seq_capture_status[1] = "NORMAL_CAPTURE"
                    log.capture_event("SEQ1", "NORMAL_CAPTURE")

        # EARLY FOLDER CREATION: allocate the panel folder as soon as
        # SEQ1 is stable for 2 frames.
        allocate_panel_folder_for_seq1(serial_present=(serial_det is not None))
    elif is_panel_seq1 and (hand_present or getattr(state, 'multiple_panels_visible', False)) and state.current_sequence in (0, 1):
        # Hand is back or multiple panels visible — reset the delay timer
        if state.seq_hand_free_since[1] is not None:
            print('[SEQ1-DELAY] Condition broken — resetting capture delay timer')
        state.seq_hand_free_since[1] = None
        state.seq1_detection_count = 0
        state.seq1_serial_stable_count = 0
        if not state.seq1_auto_captured:
            if getattr(state, 'multiple_panels_visible', False):
                state.status_msg = "SEQ1 — Lay panel flat to capture"
            else:
                state.status_msg = "SEQ1 — Remove hand for capture"
    elif not is_panel_seq1:
        # Reset counter when panel is no longer visible
        if getattr(state, 'seq1_detection_count', 0) > 0:
            print(f"[SEQ1] Lost panel — resetting stability counter")
        state.seq1_detection_count = 0
        state.seq1_serial_stable_count = 0

    # ── PORTRAIT TILT → Camera-2 OCR trigger ─────────────────────────────────
    # When the operator tilts the panel from horizontal (landscape) to vertical
    # (portrait) during SEQ1, the serial-number face is now pointing at Camera-2.
    # Trigger Camera-2 OCR immediately at this moment — earlier than wiping.
    #
    # Conditions:
    #   • SEQ1 snapshot already taken  (real panel confirmed, not empty table)
    #   • best_any_panel visible with a portrait bbox  (height > width)
    #   • current_sequence == 1  (still in SEQ1 phase)
    #   • OCR not already started
    #   • Camera-2 instance available
    _panel_is_portrait = False
    if best_any_panel is not None:
        _bx1, _by1, _bx2, _by2 = best_any_panel['bbox']
        _bw = max(1, _bx2 - _bx1)
        _bh = max(1, _by2 - _by1)
        _panel_is_portrait = (_bh > _bw * 1.1)   # clearly taller than wide

    if (_panel_is_portrait
            and state.current_sequence == 1
            and not getattr(state, 'ocr_started', False)
            and _CAM2_OCR_AVAILABLE
            and camera2_ocr_instance is not None):
        # ✅ FIX: removed seq1_snapshot_data dependency.
        # SSP_TAILGATE_SEQ_PLANT2-V1.pt has no serial_number class → serial_det always None
        # → snapshot gate always blocked → OCR never started.
        # Portrait + SEQ1 is sufficient. Camera2 SSP_TAILGATE_SERIAL.pt handles detection.
        folder = ensure_panel_folder()
        if folder:
            camera2_ocr_instance.set_panel_folder(folder)
            camera2_ocr_instance.start_ocr()
            state.ocr_started = True
            print('[CAM2-OCR] 🔄 OCR TRIGGERED by PORTRAIT TILT', flush=True)
            print(f'[CAM2-OCR]    Panel folder: {folder}', flush=True)
            print(f'[CAM2-OCR]    YOLO ready: '
                  f'{camera2_ocr_instance.serial_detector is not None}',
                  flush=True)
    elif _panel_is_portrait and state.current_sequence == 1:
        print(f'[CAM2-OCR] Portrait but not started: '
              f'ocr_started={getattr(state,"ocr_started",False)} '
              f'cam2={camera2_ocr_instance is not None}', flush=True)

    # ── 5. Update panel contour + mask from detection ─────────────
    if panel_det is not None:
        bx1, by1, bx2, by2 = panel_det['bbox']
        pad_x = int((bx2 - bx1) * 0.03)
        pad_y = int((by2 - by1) * 0.03)
        bx1 = max(0, bx1 - pad_x)
        by1 = max(0, by1 - pad_y)
        bx2 = min(w, bx2 + pad_x)
        by2 = min(h, by2 + pad_y)
        state.panel_contour  = detect_panel_contour_in_bbox(frame, (bx1, by1, bx2, by2))
        state.panel_rect     = (bx1, by1, bx2 - bx1, by2 - by1)
        state.panel_bbox_xyxy = (bx1, by1, bx2, by2)   # FIX 3: update for capture()
        pm = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(pm, [state.panel_contour], -1, 255, -1)
        state.panel_mask = pm
        state.current_panel_name = panel_det['name']
    elif best_any_panel is not None:
        # FIX 2: TRACK B (panel_det) may be None due to class-lock during transitions.
        # Update current_panel_name from TRACK A (best_any_panel) as fallback so
        # seq2_wiping_frames / seq3_wiping_frames counters are not blocked by a
        # stale class name from the previous sequence.
        #
        # Relaxed rule: if TRACK A strongly believes the panel class (confidence>=0.40)
        # then accept it as the current_panel_name even if it differs from the
        # expected class. This helps SEQ3 progress/counting during fast transitions
        # where class-locking may briefly hold back the UI's panel name.
        _bap_name = best_any_panel['name']
        _bap_conf = best_any_panel.get('confidence', 0.0)
        _expected  = f'panel_seq{state.current_sequence}'
        if _bap_name == _expected or (_bap_name in ('panel_seq2', 'panel_seq3') and _bap_conf >= 0.40):
            state.current_panel_name = _bap_name
    else:
        pass  # panel absent — current_panel_name left as-is for hold-decay logic
        
    if best_any_panel is not None:
        if getattr(state, 'panel_absent_frames', 0) > 30:
            state.was_absent_long = True
        state.panel_absent_frames = 0
    else:
        state.panel_absent_frames = getattr(state, 'panel_absent_frames', 0) + 1
        # FIX B2: force held overlay off after 8 absent frames so UI badge
        # clears immediately when panel is physically removed on Pi.
        if state.panel_absent_frames > 8:
            with _overlay_lock:
                _held_panel['ttl'] = 0
            state.model_sees = 'NONE'
    # ── 6. Stable sequence buffer ─────────────────────────────────
    conf_seq = stable_seq(raw_seq_id)

    # ── SEQ1 ─────────────────────────────────────────────────────────
    # Guard: only attempt once. If capture() fails (disk full, folder race etc.)
    # we do NOT retry every frame at 25fps — that floods logs and wastes I/O.
    # seq1_auto_captured is set True on the first attempt regardless of outcome;
    # a failure is logged clearly so the operator can investigate.
    if not state.seq1_auto_captured and state.seq1_snapshot_data is not None:
        state.seq1_auto_captured = True   # mark immediately — no repeated attempts
        ok = capture(1, state.seq1_snapshot_data['frame'], metadata=state.seq1_snapshot_data)
        if ok:
            print("[CAPTURE] SEQ1 ✅ — full frame + crops saved")
        else:
            print("[CAPTURE] SEQ1 ❌ — capture_sequence_images failed "
                  "(check folder, disk space, frame validity)")


    # ── SEQ2  ─────────────────────────────────────────────────────────────────
    # Strategy: track the BEST sharpness horizontal frame during SEQ2.
    # Capture fires once we have 3 good landscape frames — using the sharpest.
    #
    # Detection source priority:
    #   1. panel_det  (TRACK B — conf≥0.50, class-locked to panel_seq2)
    #   2. best_any_panel  (TRACK A — conf≥0.35, used when TRACK B is None)
    # Using TRACK A as fallback prevents missed captures when model outputs
    # panel_seq2 at 0.40-0.49 conf (common on Pi with RTSP compression).
    #
    # Stable counter:
    #   • Increments on every valid landscape frame
    #   • Only resets after 3 CONSECUTIVE non-landscape / no-detection frames
    #   • This tolerates 1-2 frame detection gaps without losing progress
    seq3_also_visible = any(d['name'] == 'panel_seq3' for d in seq_cands_A)

    # Best available panel_seq2 detection this frame
    _seq2_det = None
    # STRICT BLOCK: if seq3 is also visible anywhere in the frame, do NOT capture seq2
    if not seq3_also_visible:
        if panel_det is not None and panel_det['name'] == 'panel_seq2':
            _seq2_det = panel_det
        elif best_any_panel is not None and best_any_panel['name'] == 'panel_seq2':
            _seq2_det = best_any_panel

    if (state.current_sequence == 2
            and not state.seq2_auto_captured
            and not state.all_sequences_done):

        if _seq2_det is not None:
            bx1, by1, bx2, by2 = _seq2_det['bbox']
            _bw = max(1, bx2 - bx1)
            _bh = max(1, by2 - by1)
            _det_conf   = _seq2_det['confidence']
            _panel_area = _bw * _bh
            # PANEL-BBOX landscape check: panel must be wider than tall.
            # Frame is always 1280x960 (w>h always True) so using
            # frame dims never rejected portrait panels — fixed.
            _is_landscape = _bw > _bh * 1.1
            _is_big_enough = (_panel_area >= 0.15 * w * h)
            _cx = (bx1 + bx2) // 2
            _cy = (by1 + by2) // 2
            _prev = getattr(state, 'seq2_prev_center', None)
            _is_stable = True
            if _prev is not None:
                _dx = abs(_cx - _prev[0])
                _dy = abs(_cy - _prev[1])
                _is_stable = _dx < 20 and _dy < 20
            state.seq2_prev_center = (_cx, _cy)

            if _is_landscape and _is_big_enough:
                frame_sharp = cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.CV_64F).var()

                if frame_sharp >= 12:
                    # ── Layer 1: Always track best frame (even with hand) ────────
                    # Ensures hand/timeout fallbacks always have a frame to use.
                    if _det_conf >= 0.40 and frame_sharp > state.seq2_best_sharp:
                        state.seq2_best_frame = frame.copy()
                        state.seq2_best_sharp = frame_sharp
                        state.seq2_best_meta  = {
                            'rect':      (bx1, by1, _bw, _bh),
                            'bbox_xyxy': (bx1, by1, bx2, by2),
                            'contour':   getattr(state, 'panel_contour', None),
                            'serial_number': state.serial_number,
                        }
                        _htag = " (with hand)" if hand_present else ""
                        print(f"[SEQ2] Best frame updated{_htag} "
                              f"sharp={frame_sharp:.0f} conf={_det_conf:.2f}")

                    # ── Layer 2: Fire capture — no hand + stable required ────────
                    # ── Layer 2: Fire capture — no hand + stable required ────────
                    # Skip if still in grace period after prompt
                    _in_grace_period = False
                    if getattr(state, '_seq2_prompt_time', None) is not None:
                        _grace_left = _SEQ2_GRACE_PERIOD - (time.time() - state._seq2_prompt_time)
                        if _grace_left > 0:
                            print(f"[SEQ2] Grace period: {_grace_left:.1f}s remaining")
                            _in_grace_period = True

                    if not _in_grace_period and not hand_present and not getattr(state, 'multiple_panels_visible', False) and _is_stable:
                        # ── NEW: hand-free delay gate ────────────────────────
                        _delay_satisfied = False
                        if state.seq_hand_free_since[2] is not None:
                            _hand_free_elapsed = time.time() - state.seq_hand_free_since[2]
                            if _hand_free_elapsed >= CAPTURE_DELAY_SECS:
                                _delay_satisfied = True

                        if _delay_satisfied:
                            # Check horizontal movement (panel must be still, not sliding left-right)
                            _prev_seq2_cx = getattr(state, 'seq2_prev_center_x', None)
                            if _prev_seq2_cx is not None:
                                _horiz_move = abs(_cx - _prev_seq2_cx)
                                if _horiz_move <= 15:  # Max 15px horizontal movement
                                    state.seq2_miss_count = 0
                                    state.seq2_landscape_count += 1
                                else:
                                    print(f"[SEQ2] Panel moving horizontally ({_horiz_move}px) — not counting")
                                    state.seq2_landscape_count = 0
                            else:
                                state.seq2_miss_count = 0
                                state.seq2_landscape_count += 1
                            state.seq2_prev_center_x = _cx
                            print(f"[SEQ2] Landscape #{state.seq2_landscape_count} "
                                  f"sharp={frame_sharp:.0f} conf={_det_conf:.2f} "
                                  f"area={_panel_area}/{int(0.15*w*h)}")

                            _fire_threshold = 3   # 3 stable landscape frames = capture immediately
                            if state.seq2_landscape_count >= _fire_threshold:
                                # ── IMMEDIATE CAPTURE on stable landscape detection ──
                                _best = state.seq2_best_frame
                                if _best is not None:
                                    print(f'[SEQ2] ✅ {_fire_threshold} stable landscape frames — capturing NOW')
                                    capture(2, _best, metadata=state.seq2_best_meta or {})
                                    state.seq2_auto_captured = True
                                    state.seq2_capture_time = now
                                    state.seq_capture_data[2] = _best
                                else:
                                    print('[SEQ2] ⚠️ No best_frame yet — deferring 1 more frame')

                else:
                    print(f"[SEQ2] Landscape but blurry ({frame_sharp:.0f})")

            elif not _is_big_enough:
                state.seq2_miss_count = getattr(state, 'seq2_miss_count', 0) + 1
                print(f"[SEQ2] Panel too small "
                      f"({_panel_area} < {int(0.15*w*h)}) — skipping")
            elif _is_landscape and not _is_stable:
                state.seq2_landscape_count = 0
                print(f"[SEQ2] MOVING — waiting for stable position")
            else:
                state.seq2_miss_count = getattr(state, 'seq2_miss_count', 0) + 1
                if state.seq2_miss_count >= 8:
                    state.seq2_landscape_count = 0
                    state.seq2_miss_count = 0
                    print(f"[SEQ2] 8 miss frames — resetting counter ({w}\u00d7{h})")
        else:
            state.seq2_miss_count = getattr(state, 'seq2_miss_count', 0) + 1
            if state.seq2_miss_count >= 8:
                state.seq2_landscape_count = 0
                state.seq2_miss_count = 0

    # ── SEQ2 fallback: transition to SEQ3 happened but SEQ2 not captured ──────
    # STRICT: Only use tracked landscape frame — never fall back to last_clean/orig.
    if (conf_seq == 3 and state.current_sequence == 3
            and not state.seq2_auto_captured
            and not state.all_sequences_done):
        fb = state.seq2_best_frame      # landscape tracked frame ONLY
        if fb is not None:
            print("[CAPTURE] SEQ2 FALLBACK — using tracked horizontal frame")
            capture(2, fb, metadata=state.seq2_best_meta or {})
            state.seq2_auto_captured = True
            state.seq_capture_data[2] = fb
        else:
            print("[CAPTURE] SEQ2 FALLBACK ⚠️ no frame tracked — "
                  "will retry via timeout or hand fallback next")

            # CROSS-CAPTURE logic removed to enforce 15s timer logic in update_wiping

    # ── SEQ2 hand-presence fallback ───────────────────────────────────────────
    # Removed: Auto-capture is now handled by strict 15s timer in update_wiping.
    # ── SEQ2 Pi-timeout fallback ───────────────────────────────────────────────
    # Removed: Auto-capture is now handled by strict 15s timer in update_wiping.

    # ── SEQ3 ──────────────────────────────────────────────────────────────────
    # Same strategy as SEQ2: track best sharpness landscape frame, capture
    # after 3 confirmed horizontal frames using the sharpest one.
    seq2_also_visible = any(d['name'] == 'panel_seq2' for d in seq_cands_A)

    _seq3_det = None
    # STRICT BLOCK: if seq2 is also visible anywhere in the frame, do NOT capture seq3
    if not seq2_also_visible:
        if panel_det is not None and panel_det['name'] == 'panel_seq3':
            _seq3_det = panel_det
        elif best_any_panel is not None and best_any_panel['name'] == 'panel_seq3':
            _seq3_det = best_any_panel

    if (state.current_sequence == 3
            and not state.seq3_auto_captured
            and not state.all_sequences_done):
        # Ensure SEQ2 is captured and marked completed before attempting SEQ3 capture.
        if not state.completed.get(2, False):
            # STRICT GATING: Do not allow SEQ3 capture or bypass SEQ2 wiping.
            # Operator must complete SEQ2 properly.
            print('[SEQ3] SEQ2 not marked completed — waiting for operator to finish SEQ2 wiping')
            state.status_msg = 'SEQ3 detected — please finish wiping SEQ2 first'
            return
        elif (_seq3_det is not None
                and not hand_present
                and not getattr(state, 'multiple_panels_visible', False)):

            bx1, by1, bx2, by2 = _seq3_det['bbox']
            _bw = max(1, bx2 - bx1)
            _bh = max(1, by2 - by1)
            _det_conf   = _seq3_det['confidence']
            _panel_area = _bw * _bh

            # PANEL-BBOX landscape check: panel must be wider than tall.
            # Fixed: was using frame dims (w>h) which always passed.
            _is_landscape = _bw > _bh * 1.1

            # Panel size gate: real panel ≥ 15% of frame.
            # Rejects empty-table false positives (bbox usually small/noisy).
            _is_big_enough = (_panel_area >= 0.15 * w * h)

            # ── Center stability: panel must not move >20px between frames
            _cx = (bx1 + bx2) // 2
            _cy = (by1 + by2) // 2
            _prev = getattr(state, 'seq3_prev_center', None)
            _is_stable = True
            if _prev is not None:
                _dx = abs(_cx - _prev[0])
                _dy = abs(_cy - _prev[1])
                _is_stable = _dx < 20 and _dy < 20
            state.seq3_prev_center = (_cx, _cy)

            # ── NEW: hand-free delay gate ────────────────────────
            _delay_satisfied = False
            if state.seq_hand_free_since[3] is not None:
                _hand_free_elapsed = time.time() - state.seq_hand_free_since[3]
                if _hand_free_elapsed >= CAPTURE_DELAY_SECS:
                    _delay_satisfied = True

            if _delay_satisfied and _is_landscape and _is_stable and _is_big_enough:
                frame_sharp = cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.CV_64F).var()

                state.seq3_miss_count = 0

                if frame_sharp >= 12:
                    # FREEZE RULE: stop updating seq3_best_frame once wipe is
                    # clearly underway (>= 5 confirmed wipe frames, not just 1).
                    # 1-frame threshold was too aggressive — a single hand brush
                    # locked the frame before a good quality image was saved.
                    # 5 frames (~1s at 5fps) still protects against wipe-frame
                    # pollution while tolerating brief accidental touches.
                    _wipe_started = (state.seq3_wiping_frames >= 5)

                    # Conf gate lowered from 0.55 → 0.30.
                    # Initial panel detections are typically 0.35-0.50 (panel
                    # partly in frame, low contrast). With conf>=0.55, the very
                    # first clear landscape frames were silently rejected and
                    # seq3_best_frame stayed None → all fallbacks failed → SEQ3
                    # image was never saved.
                    if (not _wipe_started
                            and _det_conf >= 0.30
                            and frame_sharp > state.seq3_best_sharp):
                        state.seq3_best_frame = frame.copy()
                        state.seq3_best_sharp = frame_sharp
                        state.seq3_best_meta  = {
                            'rect':          (bx1, by1, _bw, _bh),
                            'bbox_xyxy':     (bx1, by1, bx2, by2),
                            'contour':       getattr(state, 'panel_contour', None),
                            'serial_number': state.serial_number,
                        }
                    state.seq3_landscape_count += 1
                    print(f"[SEQ3] Landscape #{state.seq3_landscape_count} "
                          f"sharp={frame_sharp:.0f} conf={_det_conf:.2f} "
                          f"area={_panel_area}/{int(0.15*w*h)} "
                          f"wipe_started={'Y' if _wipe_started else 'N'} "
                          f"stable={'Y' if _is_stable else 'N'}")

                    # Fire immediately on 1st frame if high-quality + pre-wipe.
                    _fire_threshold = (4 if (_det_conf >= 0.65 and frame_sharp >= 25   # INCREASED: need 4 frames high-quality
                                             and not _wipe_started
                                             and state.seq3_best_frame is not None)
                                       else 6)   # INCREASED: need 6 frames normal
                    if state.seq3_landscape_count >= _fire_threshold:
                        # ── GUARD: SEQ2 must be captured before SEQ3 fires ──
                        if not state.seq2_auto_captured:
                            _bf2 = state.seq2_best_frame
                            if _bf2 is not None:
                                print('[SEQ3] ⚡ SEQ2 not captured yet — saving SEQ2 now')
                                capture(2, _bf2, metadata=state.seq2_best_meta or {})
                                state.seq2_auto_captured = True
                                state.seq2_capture_time = time.time()
                            else:
                                print('[SEQ3] ⚠️ SEQ3 ready but SEQ2 has no landscape frame')
                        # ── IMMEDIATE CAPTURE on stable landscape detection ──
                        if not state.seq3_auto_captured:
                            _best3 = state.seq3_best_frame
                            if _best3 is not None:
                                print(f'[SEQ3] ✅ {_fire_threshold} stable landscape frames — capturing NOW')
                                capture(3, _best3, metadata=state.seq3_best_meta or {})
                                state.seq3_auto_captured = True
                                state.seq3_capture_time = time.time()
                                state.seq_capture_data[3] = _best3
                            else:
                                print('[SEQ3] ⚠️ No best_frame yet — deferring 1 more frame')
                else:
                    print(f"[SEQ3] Landscape but blurry ({frame_sharp:.0f})")

            elif not _is_big_enough:
                state.seq3_miss_count = getattr(state, 'seq3_miss_count', 0) + 1
                print(f"[SEQ3] Panel too small "
                      f"({_panel_area} < {int(0.15*w*h)}) — "
                      "likely empty-table false positive, skipping")

            elif _is_landscape and not _is_stable:
                state.seq3_landscape_count = 0
                print(f"[SEQ3] MOVING — waiting for stable position")

            else:
                state.seq3_miss_count = getattr(state, 'seq3_miss_count', 0) + 1
                if state.seq3_miss_count >= 3:
                    state.seq3_landscape_count = 0
                    state.seq3_miss_count       = 0
                    print(f"[SEQ3] 3 non-landscape frames — "
                          f"resetting counter (frame {w}×{h})")
        else:
            state.seq3_miss_count = getattr(state, 'seq3_miss_count', 0) + 1
            if state.seq3_miss_count >= 5:
                state.seq3_landscape_count = 0
                state.seq3_miss_count       = 0

    # ── SEQ3 Fallbacks Removed to Enforce 15s Logic in update_wiping ──

    # ── 8. State Machine & Sequence Transitions ───────────────────
    # [RULE] RESET MUST HAPPEN BEFORE update_seq() — so any new panel's
    # SEQ1 initialises with a completely clean state and doesn't inherit
    # timers, wipe%, or completed flags from the previous panel.
    #
    # Reset when panel_seq1 arrives:
    #   - Case A: all_sequences_done=True → normal end-of-cycle reset.
    #   - Case B: On SEQ2/SEQ3, panel_seq1 re-appears → panel swap.
    #             B1: was_absent_long (>30 frames) — definitive removal.
    #             B2: any absence (≥5 frames) + we are on SEQ2/SEQ3 — fast swap.
    if conf_seq == 1:
        if getattr(state, 'all_sequences_done', False):
            print("[PIPELINE] Cycle complete + panel_seq1 detected — resetting for new panel.")
            reset_panel()
        elif state.current_sequence in (2, 3):
            absent = getattr(state, 'panel_absent_frames', 0)
            was_absent_long = getattr(state, 'was_absent_long', False)
            # B1: Long absence (>30 frames) — confirmed panel removal
            if was_absent_long:
                print(f"[PIPELINE] panel_seq1 seen after long absence during SEQ{state.current_sequence}"
                      f" — new panel swap, finalising old panel then resetting.")
                _finalize_panel()  # SAVE THE OLD PANEL DATA BEFORE RESET
                reset_panel()
                state.was_absent_long = False
            # B2: Short absence (≥5 frames) while on SEQ2/SEQ3 and stable SEQ1 detected.
            # 5 frames ≈ 0.2s at 25fps — enough to confirm real removal vs model noise.
            elif absent >= 5:
                print(f"[PIPELINE] panel_seq1 seen after {absent}-frame absence during"
                      f" SEQ{state.current_sequence} — treating as fast panel swap, resetting.")
                _finalize_panel()  # Save old panel data
                reset_panel()
                state.was_absent_long = False
        elif state.current_sequence == 1:
            # SEQ1 → SEQ1 re-detection — only reset if panel was absent long
            if getattr(state, 'was_absent_long', False):
                print("[PIPELINE] panel_seq1 re-detected after long absence during SEQ1 — resetting.")
                _finalize_panel()
                reset_panel()
                state.was_absent_long = False
        # When panel is removed during SEQ3 and image is captured, complete it.
        if (state.current_sequence == 3
                and not state.completed.get(3, False)
                and getattr(state, 'seq3_auto_captured', False)
                and getattr(state, 'panel_absent_frames', 0) >= 90): # INCREASED: was 30 frames
            print('[PIPELINE] Panel removed during SEQ3 (image captured) — marking SEQ3 complete and finalizing')
            complete(3)
    elif conf_seq in (2, 3):
        state.was_absent_long = False

    # ── Now advance sequence (AFTER reset, so new panel starts clean) ───
    if conf_seq in (1, 2, 3):
        # Capture SEQ2 detection confidence when transitioning to SEQ2
        if conf_seq == 2 and state.current_sequence == 1 and best_any_panel:
            state.seq2_detection_confidence = best_any_panel.get('confidence', 0)
            print(f"[SEQ2] 📍 Detected with confidence {state.seq2_detection_confidence:.2f}")
        # Advance sequence immediately on detection for better UI response.
        update_seq(conf_seq)

    # Count ALL frames spent in SEQ2 — used by the timeout fallback.
    # FIX: moved here from update_wiping() where it only counted wiping frames.
    # This ensures the 20-frame timeout fires even before the operator starts wiping.
    if state.current_sequence == 2:
        state.seq2_in_sequence_frames = getattr(state, 'seq2_in_sequence_frames', 0) + 1

    # ── 8b. Wiping & Motion Detection ──────────────────────────────
    # Motion detection for wiping (Must happen AFTER update_seq so current_sequence is correct)
    gray, motion_mask = detect_motion(frame, getattr(state, 'prev_gray', None))
    state.prev_gray  = gray
    wiping_pixel     = detect_wiping(state.hand_bbox, motion_mask, state.panel_mask)
    motion           = (wiping_pixel is not None
                        and cv2.countNonZero(wiping_pixel) > 100)
    if motion and wiping_pixel is not None:
        if state.cleaned_mask is None or state.cleaned_mask.shape != wiping_pixel.shape:
            state.cleaned_mask = np.zeros_like(wiping_pixel)
        state.cleaned_mask = cv2.bitwise_or(state.cleaned_mask, wiping_pixel)
    update_wiping(hand_present, motion)



    # ── 9b. Precise Sequence Completion & Lead Time Logic ───────────
    # If panel was flipped to SEQ2 but SEQ1 wasn't completed (i.e. wiping wasn't 100%)
    if state.current_sequence == 2 and not state.completed.get(1):
        s1_pct  = getattr(state, 'seq1_final_wipe_pct', 0)
        s2_conf = getattr(state, 'seq2_detection_confidence', 0)

        # SEQ1 was not fully wiped before flipping
        print(f"[SEQ1] ⚠️ SEQ2 detected but SEQ1 wiping was only {s1_pct}% — marking missed")
        state.sequence_status[1] = "missed"
        state.completed[1] = True  # mark it done so we don't keep checking
        alert_missed(1)
        
        # ── Stop OCR scanning once SEQ1 is done ──────────────────────
        # Serial number is only on the SEQ1 panel face.
        if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
            if camera2_ocr_instance.is_done():
                camera2_ocr_instance._is_scanning = False
                print('[CAM2-OCR] SEQ1 missed — OCR confirmed, scanning stopped')
            else:
                print('[CAM2-OCR] SEQ1 missed — OCR still running; '
                      'keeping scanning alive to finish 3-slot collection')

    # [RULE] SEQ3 Completion
    # Problem: ghost detections after removal keep resetting seq3_absence_frames.
    # Fix A — Ghost filter: after progress>=90%, only count conf>=0.65 as "present".
    # Fix B — Panel MUST fully disappear: 30+ absence frames (~1.2s at 25fps).
    # Fix C — Timeout path also requires panel to be absent (no timeout while visible).
    #
    # Rules:
    #   • 90% wipe detected (not 85%)
    #   • Panel completely gone: seq3_absence_frames >= 30
    #   • Hand removed >= 1.5s
    if state.current_sequence == 3 and not state.completed.get(3, False):

        # Ghost confidence threshold (strict - only accept very strong detections)
        # After 90% wipe, require very high confidence (0.85) to consider panel still present
        # This ensures panel must be COMPLETELY out of frame/view
        ghost_threshold = 0.85 if state.progress >= 90 else 0.0
        seq3_really_present = (
            conf_seq == 3
            and best_any_panel is not None
            and best_any_panel.get('confidence', 0) >= ghost_threshold
        )

        if seq3_really_present:
            state.seq3_seen_frames    += 1
            state.seq3_absence_frames  = 0
        else:
            state.seq3_absence_frames += 1

        # Calculate how long the hand has been gone (must be gone to finalize)
        hand_gone_secs = (time.time() - state.hand_gone_since) if getattr(state, 'hand_gone_since', None) else 0

        # ✅ FIX: SEQ3 timeout auto-complete if operator doesn't place panel
        # Prevent system from waiting forever if operator skips SEQ3
        if (state.current_sequence == 3 
            and not state.seq3_auto_captured
            and state.seq_start_time.get(3) is not None):
            
            seq3_elapsed = time.time() - state.seq_start_time[3]
            
            # If SEQ3 has been waiting for 60+ seconds without image capture
            if seq3_elapsed >= 60.0:
                print(f"[SEQ3] ⚠️ TIMEOUT: No image captured in {seq3_elapsed:.0f}s")
                print(f"[SEQ3] 🔔 Auto-completing SEQ3 (operator didn't place panel)")
                # Force complete with timeout flag
                state.seq3_auto_captured = True
                if not hasattr(state, 'seq_capture_status'): 
                    state.seq_capture_status = {}
                state.seq_capture_status[3] = "TIMEOUT_NO_CAPTURE"
                print(f"[SEQ3] ✅ Auto-completed with status: TIMEOUT_NO_CAPTURE")
                print(f"[SEQ3] ✅ New panel can now arrive and start fresh")

            # Display progress while waiting (log every 10s)
            elif seq3_elapsed > 5.0:
                if not hasattr(state, '_seq3_last_log_time'):
                    state._seq3_last_log_time = 0
                current_log = int(seq3_elapsed / 10) * 10
                if current_log > state._seq3_last_log_time:
                    print(f"[SEQ3] ⏳ Waiting for panel ({seq3_elapsed:.0f}s elapsed, timeout at 60s)")
                    state._seq3_last_log_time = current_log

        # [RULE] Normal completion:
        #   • 90%+ wipe
        #   • Panel absent 45+ consecutive frames (~1.8s at 25fps) - fully removed from table
        #   • SEQ3 image MUST be captured (seq3_auto_captured == True)
        #   NOTE: hand_gone_secs check removed — operator's hand stays visible
        #   while carrying the panel away, preventing completion indefinitely.
        #   seq3_absence_frames >= 45 ensures panel is COMPLETELY gone and not coming back.
        normal_done = (
            state.seq3_seen_frames >= 5
            and state.seq3_absence_frames >= 45  # INCREASED: was 30, now 45 for complete removal
            and state.seq3_auto_captured  # ← MUST have captured image
        )

        # [RULE] Timeout completion:
        #   • 90%+ wipe, no wipe activity for 6s, panel absent for 45+ frames
        #   • SEQ3 image MUST be captured
        wiping_idle_secs = (time.time() - state.last_wiping_time) \
                           if state.last_wiping_time else 99
        timeout_done = (
            state.seq3_seen_frames >= 5
            and state.progress >= 90
            and wiping_idle_secs >= 6.0
            and state.seq3_absence_frames >= 45  # INCREASED: was 30, now 45 for complete removal
            and state.seq3_auto_captured  # ← MUST have captured image
        )

        if normal_done or timeout_done:
            reason = "panel fully removed" if normal_done else \
                     f"wipe inactivity ({wiping_idle_secs:.1f}s) + panel gone"
            print(f"--- [SEQ3 FINALIZED] --- {reason}, "
                  f"progress={state.progress}% "
                  f"absence={state.seq3_absence_frames} frames")
            if not state.seq3_auto_captured:
                print("--- [WARNING] --- SEQ3 image was missed!")
            complete(3)


    # ── Camera-2 OCR result check (ADD ONLY — no Camera-1 change) ───
    # Poll the OCR thread non-blockingly; update serial when ready.
    # ── Camera-2 OCR result check ─────────────────────────────────────
    if state.ocr_started and _CAM2_OCR_AVAILABLE and camera2_ocr_instance:
        cam2_serial  = camera2_ocr_instance.get_serial_number()
        cam2_partial = getattr(camera2_ocr_instance, 'partial_serial', None)
        # Update partial for live feedback (exclude non-meaningful values)
        if cam2_partial and cam2_partial not in ('Reading...', 'UNKNOWN'):
            state.partial_serial = cam2_partial
        # Update confirmed serial (only real values — not placeholders)
        # FIX: was "if not state.ocr_done" which fired once when serial='Reading...'
        # (Phase-1 just captured the frame) then BLOCKED the real serial update
        # when background OCR finished later.  Now we gate on the CURRENT
        # state.serial_number value instead — update whenever we get a real serial.
        if (camera2_ocr_instance.is_done()
                and cam2_serial
                and cam2_serial not in ('UNKNOWN', 'Reading...')):
            if state.serial_number in (None, 'UNKNOWN', 'Searching...', 'Reading...', ''):
                state.serial_number = cam2_serial
                state.ocr_done      = True
                if hasattr(state, 'seq_capture_status') and state.seq_capture_status.get(1) == "FALLBACK_NO_SERIAL":
                    state.seq_capture_status[1] = "LATE_SERIAL_BIND"
                    log.capture_event("SEQ1", "LATE_SERIAL_BIND")
                print(f'[CAM2-OCR] Serial confirmed and pushed to UI: {cam2_serial}')

    if state.ocr_done and not getattr(state, 'folder_renamed', False):
        try:
            old_path = state.current_sequence_panel_folder
            serial   = state.serial_number

            if (old_path and os.path.exists(old_path)
                    and serial and serial not in ('UNKNOWN', 'Reading...')):
                # FIX 3: include panel_id in rename target so each panel
                # gets a unique folder even if same serial appears twice/day.
                # Was: dirname/serial → collides when same serial reused same day.
                _tgt = (f"{serial}_{state.panel_id}"
                        if state.panel_id else serial)
                if os.path.basename(old_path) != _tgt:
                    new_path = os.path.join(os.path.dirname(old_path), _tgt)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        state.current_sequence_panel_folder = new_path
                        print(f"[FOLDER] Renamed → {new_path}")
                    else:
                        state.current_sequence_panel_folder = new_path

                    # BUG FIX: camera2_ocr_instance still holds the OLD folder
                    # path in .panel_folder, .cam2_raw_path, .cam2_roi_path.
                    # _background_ocr uses these — if stale it bails immediately
                    # (cv2.imread on old path → None) and ALL cam2_temp debug
                    # files + file renames never happen.
                    # Update them here so background_ocr resolves the new path.
                    if _CAM2_OCR_AVAILABLE and camera2_ocr_instance:
                        camera2_ocr_instance.panel_folder = new_path
                        for attr in ('cam2_raw_path', 'cam2_roi_path'):
                            p = getattr(camera2_ocr_instance, attr, None)
                            if p and old_path in p:
                                setattr(camera2_ocr_instance, attr,
                                        p.replace(old_path, new_path, 1))
                                print(f"[FOLDER] cam2 path updated: {attr}")

                # FIX: rename image files too — was missing, so files kept
                # "Searching..." prefix even after the folder was renamed.
                rename_panel_images(state.current_sequence_panel_folder, serial)

                state.folder_renamed = True

                # Save best Camera-2 frame to panel folder as Serial_Original.
                # Priority: cam2_raw_path (best slot frame already saved to disk)
                # → annotated frame of that moment as fallback.
                try:
                    dest       = state.current_sequence_panel_folder
                    serial_orig = os.path.join(dest,
                                               f"{serial}_Serial_Original.jpg")
                    # Use the best slot frame path if it exists on disk
                    best_path = getattr(camera2_ocr_instance,
                                        'cam2_raw_path', None)
                    if best_path and os.path.exists(best_path):
                        import shutil
                        shutil.copy2(best_path, serial_orig)
                        state.cam2_image_path = serial_orig
                        print(f"[CAM2] Serial_Original copied from best slot → "
                              f"{os.path.basename(serial_orig)}")
                    else:
                        # Fallback: latest Camera-2 frame
                        cam2_img = (camera2_ocr_instance.get_annotated_frame()
                                    if camera2_ocr_instance else None)
                        if cam2_img is not None:
                            cv2.imwrite(serial_orig, cam2_img)
                            state.cam2_image_path = serial_orig
                            print(f"[CAM2] Serial_Original saved (latest frame) → "
                                  f"{os.path.basename(serial_orig)}")
                except Exception as _ce:
                    print(f"[CAM2] Serial_Original save error: {_ce}")
        except Exception as _re:
            print(f"[FOLDER] Rename error: {_re}")

    if not getattr(state, 'cam2_frame_saved', False):
        try:
            # FIX A3: gate on active panel session — prevents premature folder
            # creation between panels when cam2_frame_saved is False after reset.
            if (_CAM2_OCR_AVAILABLE and camera2_ocr_instance
                    and state.panel_id is not None
                    and state.current_sequence > 0):
                # ── Priority 1: Tight ROI (Zoomed preferred)
                p = getattr(camera2_ocr_instance, 'cam2_roi_path', None)
                if not p:
                    p = getattr(camera2_ocr_instance, 'cam2_raw_path', None)
                
                # ── Priority 2: In-progress or failed frame (Attempt_ prefix)
                if not p or not os.path.exists(p):
                    folder = ensure_panel_folder()
                    import glob
                    # Prioritize Attempt ROI over Attempt Full
                    f = glob.glob(os.path.join(folder, "Attempt_*_Cam2_ROI_*.jpg"))
                    if not f:
                        f = glob.glob(os.path.join(folder, "Attempt_Cam2_OCR_Reference_Full_*.jpg"))
                    if f: p = f[0]
                
                if p and os.path.exists(p):
                    state.cam2_image_path = p
                    state.cam2_frame_saved = True
                    print(f"[CAM2] Linked tight ROI frame to PDF: {os.path.basename(p)}")
        except Exception as e:
            print(f"[CAM2 LINK ERROR] {e}")

    # ── 10. Debug log every 30 frames ──────────────────────────────
    state.frame_count = getattr(state, 'frame_count', 0) + 1
    if state.frame_count % 30 == 0:
        print(f"  [STATE] SEQ={state.current_sequence} "
              f"prog={state.progress}% "
              f"wipe={state.total_wipe_seconds:.1f}s "
              f"model_sees={getattr(state,'model_sees','-')} "
              f"conf_seq={conf_seq}")

    # FIX I2: model_sees uses TRACK A (what model actually outputs).
    # TRACK B (panel_det) is class-locked so it returns None during
    # misclassification — that was causing model_sees='NONE' and blank UI class.
    state.model_sees = best_any_panel['name'] if best_any_panel else "NONE"

    # ── 11. Update held annotation state (stable, no-flicker overlay) ────────
    # update_held_boxes keeps confirmed bboxes alive for PANEL_HOLD_FRAMES so
    # draw_overlay never flickers even when inference is slower than the stream.
    update_held_boxes(detections, conf_seq)
    # Keep a copy for OCR ROI-crop logic in run_ocr() — NOT used for display.
    state.latest_detections = detections

    # ── 12. Draw overlay + blue tint (on a copy for the UI, original stays clean)
    out = apply_blue_tint(frame.copy(), state.panel_mask, state.cleaned_mask)
    out = draw_overlay(out, state.current_sequence, state.status_msg, state.progress)
    return out



# ─────────────────────────────────────────────────────────────
# CAMERA STREAMING
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# VIDEO SOURCE STATE
# ─────────────────────────────────────────────────────────────
# Holds the path to an uploaded video file (None = live camera)
state.video_source = None   # set by /api/upload_video


# ─────────────────────────────────────────────────────────────
# FRAME GENERATORS
# ─────────────────────────────────────────────────────────────

def _stream(cap, native_fps=None):
    """
    4-thread pipeline:
      READER  → puts raw frames into raw_q (always latest)
      ENCODER → reads raw_q, encodes JPEG, puts into stream_q (always latest)
      INFER   → reads raw_q independently, runs inference, updates overlay
      STREAMER → yields from stream_q at full speed

    Stream is NEVER blocked by inference. Inference runs independently
    and updates a shared overlay that ENCODER applies to raw frames.
    """
    import queue as _queue

    my_generation = state.stream_generation
    running       = threading.Event()
    running.set()

    # Shared: latest raw frame and latest overlay
    raw_lock      = threading.Lock()
    raw_frame     = [None]     # latest BGR frame
    raw_frame_cnt = [0]        # monotonic counter — increments every new frame
    overlay_lock  = threading.Lock()
    overlay_frame = [None]     # latest inference result (annotated frame)

    # Output queue: encoder → streamer
    # maxsize=1: encoder NEVER blocks — always drops old frame and puts new one.
    # Old maxsize=2 caused encoder to block for up to 100ms waiting for streamer
    # to consume, which cascaded into a full pipeline freeze.
    stream_q = _queue.Queue(maxsize=1)

    # ── Thread 1: READER ─────────────────────────────────────────
    def reader():
        nonlocal cap
        source_url  = state.video_source
        is_rtsp     = isinstance(source_url, str) and source_url.startswith('rtsp://')
        is_file     = source_url and os.path.exists(str(source_url)) and not is_rtsp
        is_tls      = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
        is_ffmpeg   = getattr(cap, '__class__', None) and cap.__class__.__name__ == 'FFmpegVideoCapture'
        
        # FIX C1: On Pi, TLS-RTSP negotiation takes 10-30 s before isOpened().
        # Old code returned permanently on False, killing the reader thread.
        # Now we spin-wait up to 60 s then fall through to the reconnect loop.
        _open_wait = 0
        while not cap.isOpened() and _open_wait < 60:
            time.sleep(1.0)
            _open_wait += 1
            if _open_wait % 10 == 0:
                print(f"[READER] ⏳ Waiting for camera to open ... {_open_wait}s")
        if not cap.isOpened():
            print("[READER] Camera not open after 60 s - entering reconnect loop")
            with raw_lock: raw_frame[0] = None
            # Do NOT return - fall through to main while loop for reconnect

        # Safety cap for playback speed
        target_fps = min(native_fps, 30.0) if native_fps else 15.0
        print(f"[READER] ✅ Thread started for {source_url} (Target FPS: {target_fps})")
        
        # Ensure we don't start with a None frame
        ok, first_frame = cap.read()
        if ok:
            with raw_lock:
                raw_frame[0] = first_frame.copy()
                raw_frame_cnt[0] += 1
            
            # ✅ FIX 3: Verify camera resolution on first frame
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                print(f"[READER] 🎥 Camera resolution check:")
                print(f"[READER]    Actual: {w}x{h}")
                print(f"[READER]    Expected: 1280x960")
                if w != 1280 or h != 960:
                    print(f"[READER] ⚠️ WARNING: Resolution mismatch!")
                    print(f"[READER]    This may cause scaling → blur/quality loss")
                    print(f"[READER]    Serial detection accuracy may be affected")
                else:
                    print(f"[READER] ✅ Resolution verified: exact match")
        
        fails = 0
        last_ok = time.time()
        last_read_time = time.time()
        # FIX 2: 1s — Reconnect immediately if camera freezes for 1 second
        STALE_TIMEOUT = 1.0

        while running.is_set() and state.stream_generation == my_generation:
            # ── Playback Throttling (for video files) ──────────────────
            if is_file:
                interval = 1.0 / target_fps
                elapsed = time.time() - last_read_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
            
            last_read_time = time.time()

            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fails = 0
                    continue

                if hasattr(cap, 'isOpened') and not cap.isOpened():
                    print(f"[READER] ⚠️ cap.isOpened() false for {source_url} — reconnecting")
                    try: cap.release()
                    except: pass
                    time.sleep(0.1)  # Faster reconnect
                    cap = _open_rtsp(source_url)
                    is_tls = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
                    fails = 0
                    last_ok = time.time()
                    state.prev_gray = None
                    continue
                
                fails += 1
                if fails % 20 == 0:
                    print(f"[READER] ⚠️ {fails} consecutive read failures from {source_url}")
                time.sleep(0.05)
                # [FIX] Hold the last frame for 1 second (fails==20) before going completely black.
                if fails == 20:
                    with raw_lock: raw_frame[0] = None
                # ── Stale-frame watchdog: force reconnect if silent drop ──
                if time.time() - last_ok > STALE_TIMEOUT and not is_file:
                    print(f'[READER] ⚠️ Stream silent for {STALE_TIMEOUT}s — forcing reconnect')
                    try: cap.release()
                    except: pass
                    time.sleep(0.1)  # Faster reconnect, do not wait 1s
                    cap = _open_rtsp(source_url)
                    is_tls = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
                    fails = 0
                    last_ok = time.time()
                    # Reset prev_gray — new stream may have different resolution
                    state.prev_gray = None
                # FIX: Use _last_frame_ts staleness check instead of counting
                # False-return failures.  For TLSRTSPCapture, False is normal
                # between frames and would previously trigger a reconnect after
                # just 1.5s (30 × 50ms).  STALE_TIMEOUT is the real guard;
                # the fails counter is only a secondary safety net here.
                elif fails >= (600 if is_tls else 200):  # ~10s for non-TLS
                    if is_ffmpeg:
                        # FFmpegVideoCapture manages its own reconnect internally
                        fails = 0
                    elif is_file:
                        print(f"[READER] ❌ Permanent failure reading file {source_url}")
                        running.clear()
                        break
                    else:
                        # Only reconnect if truly stale (last good frame >5s ago).
                        # This guards against False returns that are just camera
                        # frame-rate gaps, not actual disconnections.
                        _stale = (time.time() - last_ok)
                        if _stale < 5.0:
                            fails = 0  # spurious failures — camera is alive, reset
                        else:
                            print(f'[READER] ⚠️ reconnecting RTSP — '
                                  f'last frame {_stale:.1f}s ago')
                            try:
                                cap.release()
                                time.sleep(0.5)
                            except: pass
                            cap = _open_rtsp(source_url)
                            is_tls = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
                            fails = 0
                            state.prev_gray = None
                continue
            
            fails = 0; last_ok = time.time()
            with raw_lock:
                raw_frame[0]      = frame.copy()
                raw_frame_cnt[0] += 1
            # Brief yield after a successful read.
            # Prevents CPU spin when camera delivers frames faster than
            # the encoder can consume them (e.g. GPU system).
            # 5 ms sleep caps effective reader rate at ~200fps max,
            # which is well above any real camera FPS.
            time.sleep(0.005)

        try: cap.release()
        except: pass

    # ── Thread 2: ENCODER (stream thread — never blocked by inference) ──
    def encoder():
        last_enc_time  = time.time()
        last_frame_ts  = time.time()
        MIN_INTERVAL   = 1.0 / STREAM_FPS_CAP
        STREAM_W       = STREAM_UI_WIDTH
        last_good_jpeg = [None]   # cache last good frame — served when camera drops

        while running.is_set() and state.stream_generation == my_generation:
            elapsed = time.time() - last_enc_time
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)
                continue

            with raw_lock:
                frame = raw_frame[0]

            if frame is None:
                # Camera dropped — serve last good frame frozen (no black screen)
                if last_good_jpeg[0] is not None:
                    try:
                        stream_q.get_nowait()
                    except _queue.Empty:
                        pass
                    stream_q.put_nowait(last_good_jpeg[0])
                    last_enc_time = time.time()
                time.sleep(0.1)
                continue

            last_frame_ts = time.time()
            h, w = frame.shape[:2]
            sh   = int(h * (STREAM_W / w))

            try:
                # ── FIX: Use INFER result when fresh; otherwise draw on
                #   SMALL frame (resize first, overlay second = 4× faster).
                #   Old code drew on 1920×1080 then shrunk → wasted CPU.
                #   INFER already drew the full annotated frame; reuse it.
                with overlay_lock:
                    cached = overlay_frame[0]

                if cached is not None:
                    # INFER pre-resized to STREAM_W already — use directly.
                    # Skip cv2.resize entirely: encoder is now zero-copy for
                    # fresh overlay frames.
                    small = cached
                else:
                    # INFER hasn't run yet (very first frames after camera start).
                    # Resize to STREAM_W FIRST, then draw overlay on the small
                    # canvas.  This is 4× faster than drawing on full 1920px frame.
                    # apply_blue_tint needs masks resized to match the small frame.
                    small  = cv2.resize(frame, (STREAM_W, sh),
                                        interpolation=cv2.INTER_LINEAR)
                    _pmask = state.panel_mask
                    _cmask = state.cleaned_mask
                    if _pmask is not None:
                        try:
                            _pmask = cv2.resize(_pmask, (STREAM_W, sh),
                                                interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            _pmask = None
                    if _cmask is not None:
                        try:
                            _cmask = cv2.resize(_cmask, (STREAM_W, sh),
                                                interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            _cmask = None
                    small = apply_blue_tint(small, _pmask, _cmask)
                    small = draw_overlay(small, state.current_sequence,
                                         getattr(state, 'status_msg', ''),
                                         getattr(state, 'progress', 0))

                # Add reconnecting message if camera is frozen for > 1.0s
                if time.time() - last_frame_ts >= 1.0:
                    overlay_text = "⏳ RECONNECTING... (Processing)"
                    cv2.putText(small, overlay_text, (50, sh // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(small, overlay_text, (50, sh // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 1, cv2.LINE_AA)

                _, buf = cv2.imencode('.jpg', small,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                jpg = buf.tobytes()
                last_good_jpeg[0] = jpg   # cache — served when camera drops

                # [FREEZE FIX] Non-blocking put: drain queue first, then put.
                # Never call put() (blocking) — that freezes the encoder thread
                # when the streamer is slow (browser lag, network hiccup, etc.)
                try: stream_q.get_nowait()
                except _queue.Empty: pass
                try: stream_q.put_nowait(jpg)
                except _queue.Full: pass   # streamer hasn't consumed yet — drop this frame
                last_enc_time = time.time()
            except Exception as e:
                print(f"[ENCODER ERR] {e}")

        stream_q.put(None)

    # ── Thread 3: INFERENCE (throttled to ~5 fps — never blocks stream) ──
    # CPU YOLO takes 200-400ms/frame. Run-then-sleep keeps encoder free.
    INFER_INTERVAL = 0.20   # 5 fps

    def infer():
        last_cnt = [-1]
        print('[INFER] Thread started')

        while running.is_set() and state.stream_generation == my_generation:

            with raw_lock:
                frame = raw_frame[0]
                cnt   = raw_frame_cnt[0]

            if frame is None:
                time.sleep(0.05)
                continue

            # Skip if no new frame since last run
            if cnt == last_cnt[0]:
                time.sleep(0.05)
                continue
            last_cnt[0] = cnt

            try:
                h, w = frame.shape[:2]
                # Store full-res original for PDF capture
                state.orig_frame = frame.copy()

                # 1. Resize for AI model; rescale bboxes back to full-res
                if w > MAX_PROC_WIDTH:
                    sx = MAX_PROC_WIDTH / w
                    small = cv2.resize(frame, (0, 0), fx=sx, fy=sx,
                                       interpolation=cv2.INTER_AREA)
                    detections = detect_hand(small)
                    for d in detections:
                        x1, y1, x2, y2 = d['bbox']
                        d['bbox'] = (int(x1/sx), int(y1/sx),
                                     int(x2/sx), int(y2/sx))
                        if d.get('polygon') is not None:
                            d['polygon'] = (d['polygon'] / sx).astype(np.int32)
                    # Scale factors: full_dim / small_dim
                    state._scale_x = 1.0 / sx
                    state._scale_y = 1.0 / sx
                else:
                    detections = detect_hand(frame)
                    state._scale_x = 1.0
                    state._scale_y = 1.0

                # 2. Run the full processing pipeline
                result = process_frame(frame, detections)

                # ── [DEBUG 6] Pipeline Flow Check ─────────────────────────
                if state.frame_count % 30 == 0:
                    print(f"[PIPELINE] detections passed to overlay layer: {len(detections)}")

                # 3. Send annotated result to encoder — pre-resized to
                #    STREAM_UI_WIDTH so encoder can use it with zero extra work.
                if result is not None:
                    try:
                        _fh, _fw = result.shape[:2]
                        _sh      = int(_fh * (STREAM_UI_WIDTH / _fw))
                        _small   = cv2.resize(result, (STREAM_UI_WIDTH, _sh),
                                              interpolation=cv2.INTER_LINEAR)
                        with overlay_lock:
                            overlay_frame[0] = _small
                    except Exception:
                        with overlay_lock:
                            overlay_frame[0] = result
                else:
                    with overlay_lock:
                        overlay_frame[0] = result

                # ── FIX O1: Periodic Garbage Collection ──────────────────────
                # Reduced from 100→300 frames: on Pi, gc.collect() every 100
                # frames (~4s at 25fps) caused 30-80ms pause spikes that briefly
                # froze the stream. 300 frames ≈ 12s keeps memory healthy.
                if state.frame_count % 300 == 0:
                    gc.collect()

            except Exception as e:
                import traceback
                print(f"[INFER THREAD ERROR] {e}")
                traceback.print_exc()
                time.sleep(0.1)

            # Throttle: only sleep if no new frame arrived during inference.
            # If camera delivered a new frame while we were running YOLO,
            # process it immediately (no extra latency).
            # If nothing new, sleep before next check.
            with raw_lock:
                _new_cnt = raw_frame_cnt[0]
            if _new_cnt == last_cnt[0]:
                # No new frame yet — sleep the full interval
                time.sleep(INFER_INTERVAL)
            # else: new frame arrived during inference → loop immediately

    # Start threads
    for t in [threading.Thread(target=reader,  daemon=True),
              threading.Thread(target=encoder, daemon=True),
              threading.Thread(target=infer,   daemon=True)]:
        t.start()

    # ── STREAMER (generator) ──────────────────────────────────────
    while True:
        try:
            buf = stream_q.get(timeout=0.5)
        except _queue.Empty:
            if not state.camera_active or \
               state.stream_generation != my_generation:
                break
            continue

        if buf is None:
            break

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf + b'\r\n')
        # [FREEZE FIX] No sleep here — stream_q.get() already provides
        # natural pacing because encoder only puts at STREAM_FPS_CAP rate.

    running.clear()


def _start_tls_proxy(host: str, cam_port: int, local_port: int) -> bool:
    """
    Start a local TCP proxy that wraps TLS -> plain TCP.
    Each camera gets its own dedicated local_port.
    """
    import socket as _socket, ssl as _ssl, threading as _threading
    
    # Track which ports are already handled to avoid restarting
    if not hasattr(_start_tls_proxy, '_active_ports'):
        _start_tls_proxy._active_ports = set()
    
    if local_port in _start_tls_proxy._active_ports:
        return True

    def handle(client):
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = _ssl.CERT_NONE
        try:
            raw = _socket.create_connection((host, cam_port), timeout=10)
            # Enable TCP_NODELAY and increase buffers
            raw.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            try:
                raw.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1048576)
                raw.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1048576)
            except: pass
            cam = ctx.wrap_socket(raw, server_hostname=host)
        except Exception as e:
            print(f'[proxy] cam connect failed: {e}')
            client.close(); return

        def pipe(src, dst):
            try:
                while True:
                    d = src.recv(8192)
                    if not d: break
                    dst.sendall(d)
            except Exception: pass
            try: src.close()
            except: pass
            try: dst.close()
            except: pass

        _threading.Thread(target=pipe, args=(client, cam), daemon=True).start()
        _threading.Thread(target=pipe, args=(cam, client), daemon=True).start()

    def server():
        try:
            srv = _socket.socket()
            srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            srv.bind(('127.0.0.1', local_port))
            # Increase server buffers
            try:
                srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1048576)
                srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1048576)
            except: pass
            srv.listen(10)
            print(f'[proxy] ✅ Proxy active: 127.0.0.1:{local_port} -> {host}:{cam_port}')
            while True:
                try:
                    client, _ = srv.accept()
                    client.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                    _threading.Thread(target=handle, args=(client,), daemon=True).start()
                except Exception as e:
                    print(f'[proxy] Accept error on {local_port}: {e}')
                    break
        except Exception as e:
            print(f'[proxy] ❌ Could not start proxy on {local_port}: {e}')

    _threading.Thread(target=server, daemon=True).start()
    _start_tls_proxy._active_ports.add(local_port)
    time.sleep(1.0) # Give it a full second to stabilize
    return True


# ─────────────────────────────────────────────────────────────────────
# HWACCEL DETECTION — run once at startup, shared by all FFmpeg cmds
# ─────────────────────────────────────────────────────────────────────

def _detect_ffmpeg_hwaccel():
    """
    Returns 'nvidia' | 'v4l2m2m' | 'software'.
    Called once at startup; result cached in _HW_TYPE.
    """
    try:
        if not FFMPEG_BIN:
            print('[HWACCEL] ffmpeg missing — assuming software decode')
            return 'software'
        r   = subprocess.run([FFMPEG_BIN, '-hwaccels'],
                             capture_output=True, text=True, timeout=4)
        out = r.stdout.lower()
        is_arm   = ('arm'   in platform.machine().lower() or
                    'aarch' in platform.machine().lower())
        has_cuda  = 'cuda'     in out or 'cuvid' in out
        has_v4l2  = 'v4l2m2m' in out

        if has_cuda and not is_arm:
            print('[HWACCEL] NVIDIA GPU detected → h264_cuvid ✅')
            return 'nvidia'
        if is_arm and has_v4l2:
            print('[HWACCEL] ARM Pi → software decode (v4l2m2m disabled)')
            return 'software'
    except Exception as e:
        print(f'[HWACCEL] probe failed ({e})')
    print('[HWACCEL] Software decode (libavcodec)')
    return 'software'

_HW_TYPE, _HW_V4L2 = _detect_ffmpeg_hwaccel(), False


def _hw_prefix():
    """
    Returns FFmpeg flags to insert BEFORE -i.
    NVIDIA: h264_cuvid GPU decode, output in system RAM (vf works fine).
    Other:  software decode.
    """
    if _HW_TYPE == 'nvidia':
        return ['-hwaccel', 'cuvid', '-c:v', 'h264_cuvid']
    return []


def _open_rtsp(url: str, width=1920, height=1080):
    if ':443/' in url or ':443' in url:
        return _TLSRTSPCapture(url, width=width, height=height)
    if not FFMPEG_BIN:
        print('[RTSP] ffmpeg not found — falling back to cv2.VideoCapture for RTSP')
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap is not None and cap.isOpened():
            return cap
        print('[RTSP] cv2.VideoCapture fallback failed — ensure OpenCV supports RTSP or install ffmpeg')
    return FFmpegVideoCapture(url, width=width, height=height)


class _TLSRTSPCapture:
    def isOpened(self):
        return getattr(self, '_opened', False)
    def release(self):
        self._running = False
        if self._proc:
            try: self._proc.kill()
            except: pass

    """
    RTSP over TLS with Digest auth — for CP Plus cameras (port 443).
    Also tries plain ffmpeg as fallback for other cameras.
    Mimics cv2.VideoCapture interface.
    """
    def __init__(self, url: str, width=1280, height=960):
        self._url      = url
        self._opened   = False
        self._w        = width
        self._h        = height
        self._proc     = None
        self._tls_sock = None
        self._lock     = threading.Lock()
        self._frame_q  = []
        self._thread   = None
        self._running  = False
        self._parse_url(url)
        self._start()

    def _parse_url(self, url: str):
        """Parse rtsp://user:pass@host:port/path"""
        import re
        m = re.match(
            r'rtsp://([^:@]+):([^@]*)@([^:/]+):?(\d*)(.*)', url)
        if m:
            self._user = m.group(1)
            self._pwd  = m.group(2)
            self._host = m.group(3)
            self._port = int(m.group(4)) if m.group(4) else 554
            self._path = m.group(5) or '/video/live?channel=1&subtype=0'
        else:
            m2 = re.match(r'rtsp://([^:/]+):?(\d*)(.*)', url)
            self._user = ''
            self._pwd  = ''
            self._host = m2.group(1) if m2 else '127.0.0.1'
            self._port = int(m2.group(2)) if m2 and m2.group(2) else 554
            self._path = m2.group(3) if m2 else '/video/live?channel=1&subtype=0'

    def _start(self):
        """Try direct ffmpeg rtsps first, then TLS socket fallback."""
        self._running = True   # must be True before reader threads start
        
        # 1. First try native ffmpeg with rtsps://
        if self._try_native_rtsps():
            return
            
        # 2. Try proxy ffmpeg
        if self._try_ffmpeg_direct():
            return
            
        # 3. Try python tls socket
        if self._try_tls_rtsp():
            return
            
        print("[RTSP] ❌ All methods failed")

    def _try_native_rtsps(self) -> bool:
        """Native rtsps:// in ffmpeg (skipping proxy)."""
        rtsps_url = self._url.replace("rtsp://", "rtsps://")
        
        self._dec_w = self._w
        self._dec_h = self._h
        frame_size  = self._dec_w * self._dec_h * 3

        # Pure software H264 decode — correct BGR24 output.
        # probesize/analyzeduration NOT restricted: FFmpeg needs enough bytes
        # to read H264 SPS/PPS headers or the decoder can't initialise.
        if not FFMPEG_BIN:
            print('[RTSP] ffmpeg not available — cannot start native rtsps')
            return False

        FFMPEG_EXE = shutil.which('ffmpeg') or FFMPEG_BIN
        print('FFMPEG FOUND:', shutil.which('ffmpeg'))
        print('FFMPEG_BIN:', FFMPEG_BIN)
        print('FFMPEG_EXE:', FFMPEG_EXE)

        cmd = [FFMPEG_EXE, '-y', '-loglevel', 'warning',
               '-user_agent', 'LibVLC',
               '-tls_verify', '0',
               '-rtsp_transport', 'tcp',
               '-rtsp_flags',     'prefer_tcp',
               '-fflags',         'nobuffer+discardcorrupt',
               '-flags',          'low_delay',
               '-i',              rtsps_url,
               # Limit output to 15 fps — reduces pipe backlog without losing freshness
               '-r',              '20',
               '-vf',             f'scale={self._dec_w}:{self._dec_h},format=bgr24',
               '-vcodec',         'rawvideo',
               '-f',              'rawvideo',
               '-vsync',          '0',
               '-an', '-sn', 'pipe:1']
        print('COMMAND:', cmd)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                # REDUCED: was 4 MB — caused multiple frames to buffer
                # and produced 5-20 s of visual lag.
                bufsize=0)  # Unbuffered — prevents stream stuck

            # Read first frame with 12s timeout (Windows-safe: no select())
            first_frame = [None]
            def read_first():
                try:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) == frame_size:
                        first_frame[0] = raw
                except: pass
            t = threading.Thread(target=read_first, daemon=True)
            t.start()
            t.join(timeout=12)
            
            if first_frame[0] is not None:
                raw = first_frame[0]
                self._proc   = proc
                self._opened = True
                frame = np.frombuffer(raw, dtype=np.uint8
                        ).reshape((self._dec_h, self._dec_w, 3)).copy()
                with self._lock:
                    self._frame_q = [frame]
                print(f'[RTSP] ✅ native rtsps {self._dec_w}x{self._dec_h}')
                

                threading.Thread(target=self._ffmpeg_frame_reader, daemon=True).start()
                return True
            proc.kill()
            err = 'FFmpeg error (not captured)'  # stderr=DEVNULL
            print(f'[RTSP] native rtsps failed: {err[-150:]}')
        except Exception as e:
            print(f'[RTSP] native rtsps error: {e}')
        return False


    def _try_ffmpeg_direct(self) -> bool:
        """Use local TLS proxy + ffmpeg."""
        import socket as _sock, select as _sel
        
        # [FIX] Assign unique proxy ports per camera IP to avoid collisions
        # Camera 1 (188) -> 5554, Camera 2 (192) -> 5555
        proxy_port = 5554 if '.188' in self._host else 5555
        _start_tls_proxy(self._host, 443, proxy_port)
        
        # Wait until proxy is actually listening
        for _ in range(30):
            try:
                t = _sock.create_connection(('127.0.0.1', proxy_port), timeout=0.1)
                t.close(); break
            except Exception:
                time.sleep(0.1)

        proxy_url = (f'rtsp://{self._user}:{self._pwd}@'
                     f'127.0.0.1:{proxy_port}{self._path}')

        # Use requested target resolution for this RTSP capture
        self._dec_w = self._w
        self._dec_h = self._h
        frame_size  = self._dec_w * self._dec_h * 3

        # Pure software H264 decode — correct BGR24 output.
        if not FFMPEG_BIN:
            print('[RTSP] ffmpeg not available — cannot start ffmpeg proxy')
            return False

        cmd = [FFMPEG_BIN, '-y', '-loglevel', 'warning',
               '-rtsp_transport', 'tcp',
               '-fflags',         'nobuffer+discardcorrupt',
               '-flags',          'low_delay',
               '-i',              proxy_url,
               '-r',              '20',
               '-vf',             f'scale={self._dec_w}:{self._dec_h},format=bgr24',
               '-vcodec',         'rawvideo',
               '-f',              'rawvideo',
               '-vsync',          '0',
               '-an', '-sn', 'pipe:1']
        try:
            print(f'[RTSP] launching ffmpeg proxy: {cmd}')
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                # REDUCED: was 4 MB buffer
                bufsize=0)  # Unbuffered — prevents stream stuck

            # Read first frame with 12s timeout (Windows-safe: no select())
            first_frame = [None]
            def read_first():
                try:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) == frame_size:
                        first_frame[0] = raw
                except: pass
            t = threading.Thread(target=read_first, daemon=True)
            t.start()
            t.join(timeout=12)
            
            if first_frame[0] is not None:
                raw = first_frame[0]
                self._proc   = proc
                self._opened = True
                frame = np.frombuffer(raw, dtype=np.uint8
                        ).reshape((self._dec_h, self._dec_w, 3)).copy()
                with self._lock:
                    self._frame_q = [frame]
                print(f'[RTSP] ✅ ffmpeg proxy {self._dec_w}x{self._dec_h} ~14fps')
                

                threading.Thread(target=self._ffmpeg_frame_reader, daemon=True).start()
                return True
            proc.kill()
            err = 'FFmpeg error (not captured)'  # stderr=DEVNULL
            print(f'[RTSP] ffmpeg proxy failed: {err[-150:]}')
        except Exception as e:
            print(f'[RTSP] ffmpeg proxy error: {e}')
        return False

    def _ffmpeg_frame_reader(self):
        """Read BGR frames from ffmpeg — with timeout protection for Windows pipe blocking."""
        frame_size = self._dec_w * self._dec_h * 3
        _buf = bytearray(frame_size)
        _mv  = memoryview(_buf)
        
        # Track frame arrival for watchdog
        _last_frame_time = [time.time()]
        _frame_timeout = 3.0  # Force reconnect if no frame in 3 seconds
        
        def watchdog_timer():
            """Monitor frame arrival, kill proc if stuck."""
            while self._running and self._proc:
                now = time.time()
                if now - _last_frame_time[0] > _frame_timeout:
                    print(f'[RTSP] ⚠️ WATCHDOG: No frame for {_frame_timeout}s — killing stuck proc')
                    try:
                        if self._proc:
                            self._proc.terminate()
                            try:
                                self._proc.wait(timeout=2)
                            except:
                                self._proc.kill()
                    except:
                        pass
                    return
                time.sleep(0.5)
        
        # Start watchdog thread
        try:
            watchdog = threading.Thread(target=watchdog_timer, daemon=True)
            watchdog.start()
        except Exception as e:
            print(f'[RTSP] watchdog error: {e}')

        while self._running and self._proc:
            try:
                # Read exactly one full frame with timeout protection
                n = 0
                _read_start = time.time()
                _read_timeout = 2.5  # 2.5s to read one full frame max
                
                try:
                    while n < frame_size and self._running and self._proc:
                        # Check timeout on each iteration
                        if time.time() - _read_start > _read_timeout:
                            print(f'[RTSP] ⚠️ TIMEOUT: Frame read stalled ({_read_timeout}s) at {n}/{frame_size} bytes')
                            break
                        
                        # Try readinto with very short attempt
                        try:
                            got = self._proc.stdout.readinto(_mv[n:])
                        except Exception as e:
                            print(f'[RTSP] readinto error: {e}')
                            break
                        
                        if not got:
                            break
                        n += got
                        
                except Exception as e:
                    print(f'[RTSP] frame read exception: {e}')
                    break

                if n != frame_size:
                    print(f'[RTSP] Incomplete frame — got {n}/{frame_size} bytes, skipping')
                    continue

                # Update frame and watchdog timer
                frame = np.frombuffer(_buf, dtype=np.uint8
                        ).reshape((self._dec_h, self._dec_w, 3)).copy()
                with self._lock:
                    self._frame_q = [frame]   # always replace — never queue
                    self._last_frame_ts = time.time()
                
                _last_frame_time[0] = time.time()  # Update watchdog timer
                
            except Exception as e:
                print(f'[RTSP] frame reader fatal error: {e}')
                break
        
        print('[RTSP] frame reader stopped')

    def _make_tls_socket(self, port):
        """Create TLS socket to camera."""
        import ssl, socket as _socket
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        raw = _socket.create_connection((self._host, port), timeout=8)
        return ctx.wrap_socket(raw, server_hostname=self._host)

    def _digest_auth(self, method, url, realm, nonce):
        import hashlib
        def md5(s): return hashlib.md5(s.encode()).hexdigest()
        ha1 = md5(f'{self._user}:{realm}:{self._pwd}')
        ha2 = md5(f'{method}:{url}')
        resp = md5(f'{ha1}:{nonce}:{ha2}')
        return (f'Digest username="{self._user}", realm="{realm}", '
                f'nonce="{nonce}", uri="{url}", response="{resp}"')

    def _try_tls_rtsp(self) -> bool:
        """RTSP-over-TLS on port 443 with Digest auth. Proven working."""
        import re, time, base64
        rtsp_url = 'rtsp://' + self._host + ':554' + self._path
        try:
            s = self._make_tls_socket(443)

            # OPTIONS -> get realm/nonce
            s.send(('OPTIONS ' + rtsp_url + ' RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: LibVLC\r\n\r\n').encode())
            time.sleep(0.3)
            r1 = s.recv(4096).decode('utf-8', errors='ignore')
            rm = re.search(r'realm="([^"]+)"', r1)
            nm = re.search(r'nonce="([^"]+)"', r1)
            if not rm: s.close(); return False
            realm = rm.group(1); nonce = nm.group(1)

            # DESCRIBE -> SDP + SPS/PPS
            auth = self._digest_auth('DESCRIBE', rtsp_url, realm, nonce)
            s.send(('DESCRIBE ' + rtsp_url + ' RTSP/1.0\r\nCSeq: 2\r\nUser-Agent: LibVLC\r\n' +
                    'Authorization:' + auth + '\r\nAccept: application/sdp\r\n\r\n').encode())
            time.sleep(0.3)
            r2 = s.recv(8192).decode('utf-8', errors='ignore')
            if '200 OK' not in r2: s.close(); return False

            spspps = b''
            fmtp = re.search(r'sprop-parameter-sets=([^;\s\r\n]+)', r2)
            if fmtp:
                for nb64 in fmtp.group(1).split(','):
                    try: spspps += b'\x00\x00\x00\x01' + base64.b64decode(nb64 + '==')
                    except: pass
            self._spspps = spspps
            print('[RTSP] ✅ TLS DESCRIBE 200 OK — SPS/PPS ' + str(len(spspps)) + ' bytes')

            # SETUP
            track_m   = re.search(r'a=control:(trackID=\d+)', r2)
            track     = track_m.group(1) if track_m else 'trackID=0'
            setup_url = rtsp_url + '/' + track
            auth3 = self._digest_auth('SETUP', setup_url, realm, nonce)
            s.send(('SETUP ' + setup_url + ' RTSP/1.0\r\nCSeq: 3\r\nUser-Agent: LibVLC\r\n' +
                    'Authorization:' + auth3 + '\r\n' +
                    'Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n').encode())
            time.sleep(0.3)
            r3 = s.recv(4096).decode('utf-8', errors='ignore')
            sm = re.search(r'Session:\s*([^\r\n;]+)', r3)
            session = sm.group(1).strip() if sm else ''
            print('[RTSP] SETUP: ' + r3.split('\n')[0].strip())

            # PLAY
            auth4 = self._digest_auth('PLAY', rtsp_url, realm, nonce)
            s.send(('PLAY ' + rtsp_url + ' RTSP/1.0\r\nCSeq: 4\r\nUser-Agent: LibVLC\r\n' +
                    'Authorization:' + auth4 + '\r\n' +
                    'Session: ' + session + '\r\nRange: npt=0.000-\r\n\r\n').encode())
            time.sleep(0.3)
            r4 = s.recv(4096).decode('utf-8', errors='ignore')
            if '200 OK' not in r4: s.close(); return False

            print('[RTSP] ✅ PLAY started — reading RTP over TLS socket')
            self._tls_sock = s
            self._opened   = True
            self._proc     = None
            self._running  = True
            self._thread   = threading.Thread(target=self._rtp_reader, daemon=True)
            self._thread.start()

            for _ in range(100):
                time.sleep(0.1)
                with self._lock:
                    if self._frame_q:
                        print('[RTSP] ✅ First frame received!')
                        return True
            print('[RTSP] ⚠️  PLAY ok but no frames yet — continuing')
            return True
        except Exception as e:
            print('[RTSP] TLS attempt failed: ' + str(e))
            return False

    def _rtp_reader(self):
        """RTP over TLS → H264 annexb → ffmpeg → BGR frames at real-time speed."""
        import struct, queue as _q

        spspps  = getattr(self, '_spspps', b'')
        dec_w   = self._w
        dec_h   = self._h

        if not FFMPEG_BIN:
            print('[RTSP] Cannot start ffmpeg RTP decoder: ffmpeg not found')
            self._running = False
            return

        cmd = [FFMPEG_BIN, '-y', '-loglevel', 'warning',
             # probesize NOT restricted (was 32) — must read SPS/PPS headers
             '-flags', 'low_delay',
             '-fflags', 'nobuffer+discardcorrupt',
             '-f', 'h264', '-i', 'pipe:0',
             '-vf', f'scale={dec_w}:{dec_h},format=bgr24',
             '-f', 'rawvideo', '-pix_fmt', 'bgr24',
             '-vsync', 'drop',
             'pipe:1']
        print(f'[RTSP] launching RTP ffmpeg decoder: {cmd}')
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Stderr reader thread for debugging
        def stderr_reader():
            while self._running:
                try:
                    line = proc.stderr.readline()
                    if not line: break
                    print(f'[RTSP-FFMPEG] {line.decode("utf-8", errors="ignore").strip()}')
                except: break
        threading.Thread(target=stderr_reader, daemon=True).start()

        frame_size = dec_w * dec_h * 3

        # Frame reader thread — continuously reads decoded BGR frames (Windows-compatible: no select())
        def frame_reader():
            last_data = [time.time()]
            WATCHDOG  = 12.0          # restart if no frame for 12s (give time for IDR)
            while self._running:
                try:
                    # Read exactly one frame — blocking read without select() (Windows-safe)
                    raw = proc.stdout.read(frame_size)
                    if len(raw) == frame_size:
                        last_data[0] = time.time()  # reset watchdog
                        frame = np.frombuffer(
                            raw, dtype=np.uint8
                        ).reshape((dec_h, dec_w, 3)).copy()
                        with self._lock:
                            self._frame_q = [frame]   # keep only latest
                            self._last_frame_ts = time.time()
                        if not hasattr(self, '_ffl'):
                            self._ffl = True
                            print('[RTSP] ✅ SUCCESS: First frame decoded and ready!')
                    elif len(raw) > 0:
                        # Partial read — keep waiting or restart
                        if time.time() - last_data[0] > WATCHDOG:
                            print('[RTSP] frame_reader: partial read watchdog — restarting')
                            proc.kill()
                            break
                    else:
                        # EOF reached
                        if time.time() - last_data[0] > WATCHDOG:
                            print('[RTSP] frame_reader: no data watchdog — restarting')
                            proc.kill()
                            break
                        time.sleep(0.1)
                except Exception as _e:
                    print(f'[RTSP] frame_reader error: {_e}')
                    break

        threading.Thread(target=frame_reader, daemon=True).start()

        # Global NAL buffer across RTP packets
        h264_buf = bytearray()
        nal_buf  = bytearray()
        buf      = b''

        # ── Send SPS/PPS first ──────────────────────────────────────
        # Crucial for ffmpeg to start decoding immediately
        if spspps:
            try:
                proc.stdin.write(spspps)
                proc.stdin.flush()
                print(f'[RTSP] Injected SPS/PPS ({len(spspps)} bytes) into decoder')
            except Exception as e:
                print(f'[RTSP] Failed to inject SPS/PPS: {e}')

        while self._running:
            try:
                self._tls_sock.settimeout(5)
                chunk = self._tls_sock.recv(131072)
                if not chunk:
                    break
                buf += chunk

                while len(buf) >= 4:
                    if buf[0] != 0x24:
                        idx = buf.find(b'$')
                        buf = buf[idx:] if idx >= 0 else b''
                        break
                    ln = struct.unpack('!H', buf[2:4])[0]
                    if len(buf) < 4 + ln:
                        break
                    ch = buf[1]; rtp = buf[4:4+ln]; buf = buf[4+ln:]
                    # Process video channels (usually 0, 2, or 4)
                    if ch % 2 != 0 or len(rtp) < 13:
                        continue

                    payload = rtp[12:]
                    marker  = (rtp[1] & 0x80) != 0
                    nt      = payload[0] & 0x1F

                    if nt == 28:   # FU-A fragmented
                        if len(payload) < 2: continue
                        fu = payload[1]
                        if fu & 0x80: # Start bit
                            nal_buf = bytearray([(payload[0] & 0xE0) | (fu & 0x1F)])
                        nal_buf.extend(payload[2:])
                        if fu & 0x40: # End bit
                            h264_buf.extend(b'\x00\x00\x00\x01' + nal_buf)
                            nal_buf = bytearray()
                    elif nt == 24:  # STAP-A (aggregated)
                        off = 1
                        while off + 2 <= len(payload):
                            sz = struct.unpack('!H', payload[off:off+2])[0]; off += 2
                            h264_buf.extend(b'\x00\x00\x00\x01' + payload[off:off+sz])
                            off += sz
                    else: # Single NAL
                        h264_buf.extend(b'\x00\x00\x00\x01' + payload)

                    # Flush to ffmpeg if we have any NALs or the marker bit is set
                    if len(h264_buf) > 0:
                        try:
                            proc.stdin.write(bytes(h264_buf))
                            proc.stdin.flush()
                            h264_buf.clear()
                        except Exception:
                            break

            except Exception as e:
                if self._running:
                    print('[RTSP] RTP reader: ' + str(e))
                break

        # Flush remaining
        try:
            if h264_buf:
                proc.stdin.write(bytes(h264_buf))
            proc.stdin.close()
            proc.kill()
        except Exception:
            pass

    def _decode_nal(self, nal_data: bytes):
        pass

    def _try_ffmpeg(self) -> bool:
        """Plain ffmpeg pipe."""
        return self._launch_ffmpeg(self._url, tls=False)

    def _launch_ffmpeg(self, url: str, tls: bool) -> bool:
        """Launch ffmpeg subprocess pipe and verify first frame."""
        if not FFMPEG_BIN:
            print('[RTSP] ffmpeg not available — cannot open ffmpeg pipe')
            return False

        cmd = [FFMPEG_BIN, '-loglevel', 'error']
        if tls:
            cmd += ['-tls_verify', '0']
        cmd += [
            '-rtsp_transport', 'tcp',
            '-i', url,
            '-vf', f'scale={self._w}:{self._h}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-an', '-sn', 'pipe:1'
        ]
        try:
            print(f'[RTSP] launching ffmpeg pipe: {cmd}')
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=10**8)
            frame_size = self._w * self._h * 3
            raw = proc.stdout.read(frame_size)
            if len(raw) == frame_size:
                self._proc    = proc
                self._buf     = raw
                self._opened  = True
                print(f"[RTSP] ✅ ffmpeg pipe {'TLS' if tls else 'plain'} — "
                      f"{self._w}×{self._h}")
                return True
            err = proc.stderr.read(300).decode('utf-8', errors='ignore')
            print(f"[RTSP] ffmpeg {'TLS' if tls else 'plain'} failed: "
                  f"{err[-100:]}")
            proc.kill()
        except Exception as e:
            print(f"[RTSP] ffmpeg error: {e}")
        return False

    def isOpened(self):
        if self._tls_sock is not None:
            return self._opened
        return self._opened and self._proc is not None

    def read(self):
        if not self.isOpened():
            return False, None
        # Both TLS and direct paths store frames in _frame_q
        with self._lock:
            if self._frame_q:
                frame = self._frame_q[-1]
                self._frame_q = []   # clear so next read() returns False until new frame
                return True, frame.copy()
        return False, None

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:          return 25.0
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._h)
        return 0.0

    def set(self, prop_id, value):
        return False

    def release(self):
        self._running = False
        if self._tls_sock:
            try: self._tls_sock.close()
            except Exception: pass
            self._tls_sock = None
        if self._proc:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception: pass
            self._proc = None
        self._opened = False


class FFmpegVideoCapture:
    def isOpened(self):
        return True # Process is managed internally
    def release(self):
        pass

    def __init__(self, url, width=1920, height=1080):   # FIXED: was height=720
        self.url = url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        
        import threading
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def _capture_loop(self):
        import subprocess
        import numpy as np
        import time
        import select

        # Pure software H264 decode — correct BGR24 output.
        # hwaccel REMOVED: v4l2m2m output format caused BGR conversion errors.
        # probesize/analyzeduration NOT restricted (were 32/0 → decoder init fail).
        if not FFMPEG_BIN:
            print('[FFmpeg Capture] ffmpeg not found — cannot start FFmpeg capture')
            time.sleep(1)
            return

        cmd_base = [FFMPEG_BIN,
                    # FIX 1: reconnect + 3s socket timeout (was 5s)
                    # -stimeout prevents readinto() blocking forever
                    # -reconnect lets FFmpeg self-recover on drops
                    '-reconnect',          '1',
                    '-reconnect_streamed', '1',
                    '-reconnect_delay_max','2',
                    '-stimeout',           '3000000',  # ✅ REDUCED to 3s
                    '-thread_queue_size',  '512',      # ✅ INCREASED (was 64)
                    '-fflags',             'nobuffer+discardcorrupt',  # ✅ ADDED discardcorrupt
                    '-flags',              'low_delay',
                    '-rtsp_transport',     'tcp']

        cmd_tail = [
            '-i',       self.url,
            '-f',       'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s',       f'{self.width}x{self.height}',
            '-r',       '10',  # ✅ FIXED: 10 fps (was 20, camera is 10)
            '-vsync',   '0',   # ✅ ADDED: Drop frames immediately
            '-an', '-sn', '-dn',
            'pipe:1',
        ]

        cmd = cmd_base + cmd_tail

        # Pre-allocate read buffer
        _buf = bytearray(self.frame_size)
        _mv  = memoryview(_buf)
        
        while self.running:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0)  # ✅ CRITICAL FIX: 0 = UNBUFFERED (was frame_size=6MB!)
                
                # Drain stderr in background so it never blocks stdout
                def _drain(p):
                    try:
                        for raw in p.stderr:
                            line = raw.decode('utf-8', errors='ignore').strip()
                            if line:
                                print(f'[CAM1-FFMPEG] {line}')
                    except Exception:
                        pass
                threading.Thread(target=_drain, args=(proc,), daemon=True).start()
                
                # ✅ FIX 2A: DEBUG - Verify process started successfully
                import time as time_module
                time_module.sleep(0.2)  # Give FFmpeg 200ms to start
                if proc.poll() is not None:
                    stderr_content = ""
                    try:
                        stderr_content = proc.stderr.read(500).decode('utf-8', errors='ignore') if proc.stderr else ""
                    except:
                        pass
                    print(f"[CAM1] ❌ FFmpeg process failed immediately!")
                    if stderr_content:
                        print(f"[CAM1] Error: {stderr_content}")
                    time.sleep(1)
                    continue  # Retry connection
                else:
                    print(f"[CAM1] ✅ Process {proc.pid} started successfully")
                
                # ✅ ADD WATCHDOG THREAD
                _last_frame_time = [time.time()]
                _frame_timeout = 3.0
                _proc_stuck = [False]
                
                def _watchdog():
                    while self.running and not _proc_stuck[0]:
                        time.sleep(0.5)
                        if proc.poll() is not None:
                            print("[CAM1] ⚠️ WATCHDOG: Process died")
                            _proc_stuck[0] = True
                            return
                        elapsed = time.time() - _last_frame_time[0]
                        if elapsed > _frame_timeout:
                            print(f"[CAM1] ⚠️ WATCHDOG: No frame for {elapsed:.1f}s - RECONNECTING")
                            _proc_stuck[0] = True
                            try:
                                proc.terminate()
                                proc.wait(timeout=2)
                            except:
                                try:
                                    proc.kill()
                                except:
                                    pass
                            return
                
                threading.Thread(target=_watchdog, daemon=True, name="CAM1-Watchdog").start()
                print(f"[CAM1] ✅ Watchdog thread started (daemon=True, timeout={_frame_timeout}s)")

            except Exception as e:
                print(f"[FFmpeg Capture] Failed to start: {e}")
                time.sleep(1)
                continue
                
            failures = 0
            while self.running and not _proc_stuck[0]:
                try:
                    # Read frame sequentially with timeout per chunk
                    n = 0
                    _read_start = time.time()
                    _read_timeout = 5.0  # ✅ 5s per frame read
                    
                    while n < self.frame_size and self.running and not _proc_stuck[0]:
                        # ✅ Check if process died
                        if proc.poll() is not None:
                            print(f"[CAM1] ⚠️ Process died")
                            _proc_stuck[0] = True
                            failures = 999
                            break
                        
                        # ✅ Check timeout
                        elapsed = time.time() - _read_start
                        if elapsed > _read_timeout:
                            print(f"[CAM1] ⚠️ Frame read timeout at {n}/{self.frame_size}")
                            failures += 1
                            break
                        
                        # ✅ Read chunk (64KB)
                        try:
                            chunk_size = min(65536, self.frame_size - n)
                            got = proc.stdout.readinto(_mv[n:n+chunk_size])
                            if not got:
                                print(f"[CAM1] ⚠️ EOF at {n}/{self.frame_size}")
                                _proc_stuck[0] = True
                                failures = 999
                                break
                            n += got
                        except Exception as e:
                            print(f"[CAM1] ❌ Read error: {e}")
                            failures += 1
                            break
                except Exception as e:
                    print(f"[CAM1] ❌ Error: {e}")
                    failures += 1
                    n = 0
                    
                if n != self.frame_size:
                    failures += 1
                    # ✅ FASTER reconnect: 3 bad frames (was 30!)
                    if failures >= 3:
                        print(f"[CAM1] {failures} bad frames → reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                    
                failures = 0
                _last_frame_time[0] = time.time()  # ✅ Update watchdog
                # FIX 5: copy frame OUTSIDE the lock — only the pointer
                # swap happens inside. Was: copy inside lock (blocked READER
                # and ENCODER for the duration of a 6MB memcopy per frame).
                frame = np.frombuffer(_buf, dtype=np.uint8
                                     ).reshape((self.height, self.width, 3)).copy()
                with self.lock:
                    self.latest_frame   = frame          # already a fresh copy
                    self._last_frame_ts = time.time()
                    self._frame_id      = getattr(self, '_frame_id', 0) + 1
                    
            # ✅ Cleanup
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                except:
                    pass
            
            if self.running:
                if _proc_stuck[0]:
                    print("[CAM1] 🔄 Watchdog triggered reconnect")
                else:
                    print("[CAM1] ⚠️ Loop exited unexpectedly - reconnecting...")
                # ✅ FASTER reconnect: 0.5s (was 1.0s)
                print("[CAM1] 🔄 Waiting 0.5s before reconnect...")
                time.sleep(0.5)

    def read(self):
        # ✅ FIX 2C: Use timeout to prevent deadlock
        acquired = self.lock.acquire(timeout=1.0)
        if not acquired:
            print("[CAM1] ⚠️ DEADLOCK DETECTED: read() lock timeout - frame skipped!")
            return False, None
        
        try:
            if self.latest_frame is None:
                return False, None
            if time.time() - getattr(self, '_last_frame_ts', time.time()) > 8.0:  # FIX 3
                return False, None
            # Only return True when a genuinely NEW frame arrived (prevents READER
            # from spinning at CPU-max speed copying the same 6 MB frame repeatedly).
            cur_id   = getattr(self, '_frame_id', 0)
            last_id  = getattr(self, '_last_read_id', -1)
            if cur_id == last_id:
                return False, None          # same frame as last read — tell caller to wait
            self._last_read_id = cur_id
            # FIX 6: no second .copy() — _capture_loop already produced
            # a fresh copy outside the lock. Returning the same object
            # is safe because _capture_loop always creates a NEW frame
            # object on every iteration (never mutates latest_frame).
            return True, self.latest_frame
        finally:
            self.lock.release()
            
    def release(self):
        self.running = False


def generate_frames():
    """
    Dual pipeline for RTSP cameras:
    - FFmpeg Hardware decodes to BGR directly into a shared variable.
    - _stream runs the reader/encoder/infer logic using the latest frame.
    """
    if state.video_source:
        source_str = str(state.video_source)
        print(f"[STREAM] Initializing video source: {source_str}")
        is_rtsp = source_str.startswith('rtsp://')
        if is_rtsp:
            cap = _open_rtsp(source_str)
            yield from _stream(cap, native_fps=25.0)
            return
        if os.path.exists(source_str):
            cap = cv2.VideoCapture(source_str)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            yield from _stream(cap, native_fps=fps)
            return

    # Local camera
    cam_id = getattr(state, 'local_camera_id', 0)
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    yield from _stream(cap, native_fps=30.0)


def _stream_rtsp_direct(rtsp_url: str):
    """
    Real-time RTSP streaming with Frame ID sync.
    Each frame gets a unique ID. Inference result is bound to
    the exact frame it was computed from. Streamer draws the
    overlay that belongs to the frame being shown.
    """
    my_gen = state.stream_generation

    # Parse URL
    import re as _re
    m = _re.match(r'rtsp://([^:@]+):([^@]*)@([^:/]+):?(\d*)(.*)', rtsp_url)
    if not m:
        print(f'[STREAM] Invalid RTSP URL: {rtsp_url}')
        return
    user, pwd, host, port, path = m.groups()

    # Start TLS proxy
    _start_tls_proxy(host, 443, 5554)
    import socket as _ck
    for _ in range(30):
        try:
            t = _ck.create_connection(('127.0.0.1', 5554), timeout=0.1)
            t.close(); break
        except: time.sleep(0.1)

    proxy_url = f'rtsp://{user}:{pwd}@127.0.0.1:5554{path}'

    # ── Frame store: {frame_id: jpeg_bytes} ──────────────────────
    import threading as _th
    frame_lock    = _th.Lock()
    frame_id      = [0]          # monotonic counter
    frame_store   = {}           # frame_id → jpeg bytes
    overlay_store = {}           # frame_id → annotated jpeg bytes
    MAX_STORE     = 5            # keep last N frames only

    # ── READER: ffmpeg → parse JPEG → store with ID ──────────────
    ffmpeg_exe = FFMPEG_BIN or shutil.which('ffmpeg')
    if not ffmpeg_exe:
        print('[STREAM] ffmpeg not found for stream capture')
        ffmpeg_exe = 'ffmpeg'
    stream_cmd = [
        ffmpeg_exe, '-y', '-loglevel', 'error',
        '-fflags', 'nobuffer+discardcorrupt',
        '-flags', 'low_delay',
        '-rtsp_transport', 'tcp',
        '-i', proxy_url,
        '-vf', 'scale=960:540',    # [FREEZE FIX] 1280:720→960:540: 44% fewer pixels
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-q:v', '4',               # [FREEZE FIX] quality 3→4: slightly smaller frames
        'pipe:1'
    ]

    def _start_ffmpeg():
        try:
            proc = subprocess.Popen(
                stream_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0)
            # Drain stderr to prevent pipe deadlock
            import threading as _th2
            def _drain(p):
                try:
                    while p.poll() is None:
                        p.stderr.readline()
                except Exception: pass
            _th2.Thread(target=_drain, args=(proc,), daemon=True).start()
            return proc
        except Exception as e:
            print(f'[STREAM] ffmpeg start failed: {e}')
            return None

    stream_proc = _start_ffmpeg()
    if stream_proc is None:
        return

    def jpeg_reader():
        nonlocal stream_proc
        buf          = b''
        last_data_ts = [time.time()]
        WATCHDOG_SEC = 15.0  # Increased from 5.0s to handle heavy CPU load

        while state.camera_active and state.stream_generation == my_gen:
            try:
                # Windows pipes are not selectable. Use a blocking read
                # with a reasonable chunk size and watchdog to detect stalls.
                try:
                    chunk = stream_proc.stdout.read(65536)
                except Exception as _e:
                    print(f'[STREAM] ffmpeg read error — restarting: {_e}')
                    try: stream_proc.kill()
                    except: pass
                    time.sleep(1.0)
                    stream_proc = _start_ffmpeg()
                    if stream_proc is None: break
                    buf = b''
                    last_data_ts[0] = time.time()
                    continue

                if not chunk:
                    print('[STREAM] ffmpeg pipe closed — restarting')
                    try: stream_proc.kill()
                    except: pass
                    time.sleep(1.0)
                    stream_proc = _start_ffmpeg()
                    if stream_proc is None: break
                    buf = b''
                    last_data_ts[0] = time.time()
                    continue

                last_data_ts[0] = time.time()
                buf += chunk

                while True:
                    s = buf.find(b'\xff\xd8')
                    if s < 0: buf = b''; break
                    e = buf.find(b'\xff\xd9', s + 2)
                    if e < 0: buf = buf[s:]; break
                    jpeg = buf[s:e + 2]
                    buf  = buf[e + 2:]
                    with frame_lock:
                        fid = frame_id[0] + 1
                        frame_id[0] = fid
                        frame_store[fid] = jpeg
                        for old in list(frame_store.keys()):
                            if old < fid - MAX_STORE:
                                frame_store.pop(old, None)
                                overlay_store.pop(old, None)
            except Exception as _ex:
                print(f'[STREAM] jpeg_reader error: {_ex}')
                time.sleep(0.5)

        try: stream_proc.kill()
        except: pass
        print('[STREAM] jpeg_reader exited')

    _th.Thread(target=jpeg_reader, daemon=True).start()

    # ── INFER: decode JPEG → Hailo → build overlay → store with same ID ─
    def infer_thread():
        last_inferred = [0]
        while state.camera_active and state.stream_generation == my_gen:
            try:
                # Get latest unprocessed frame
                with frame_lock:
                    fid  = frame_id[0]
                    jpeg = frame_store.get(fid)

                if jpeg is None or fid == last_inferred[0]:
                    time.sleep(0.04)
                    continue

                last_inferred[0] = fid

                # Decode to BGR
                arr   = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Run inference — updates state, returns annotated frame
                h_fr, w_fr = frame.shape[:2]
                state.orig_frame = frame.copy()
                state._scale_x   = 1.0
                state._scale_y   = 1.0
                detections = detect_hand(frame)
                result = process_frame(frame, detections)

                # Store clean frame for capture
                if state.hand_bbox is None and state.panel_rect is not None:
                    state.last_clean_frame      = frame.copy()
                    state.last_clean_panel_rect = state.panel_rect
                    state.last_clean_contour    = state.panel_contour
                    state.last_clean_serial_det = state.serial_det

                # Draw overlay on THIS exact frame — bind result to frame ID
                annotated = result if result is not None else draw_overlay(frame.copy(), state.current_sequence, getattr(state, 'status_msg', ''), getattr(state, 'progress', 0))
                _, buf = cv2.imencode('.jpg', annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                with frame_lock:
                    overlay_store[fid] = buf.tobytes()
                    # Evict old overlays
                    for old in list(overlay_store.keys()):
                        if old < fid - MAX_STORE:
                            overlay_store.pop(old, None)

                # ~5fps inference rate
                time.sleep(0.2)

            except Exception as e:
                print(f'[INFER] {e}')
                time.sleep(0.1)

    _th.Thread(target=infer_thread, daemon=True).start()

    # ── STREAMER: serve latest frame — no freeze, no flicker ─────
    MIN_INTERVAL   = 1.0 / STREAM_FPS_CAP
    last_streamed_fid = [0]
    last_overlay   = [None]
    last_overlay_fid = [0]

    while state.camera_active and state.stream_generation == my_gen:
        with frame_lock:
            fid = frame_id[0]
            raw = frame_store.get(fid)
            # Update last overlay if new one available
            for check in range(fid, max(0, fid - MAX_STORE), -1):
                if check in overlay_store:
                    if check > last_overlay_fid[0]:
                        last_overlay[0]     = overlay_store[check]
                        last_overlay_fid[0] = check
                    break

        if raw is None:
            time.sleep(0.005)
            continue

        # Always stream latest raw frame — overlay applied on top if available
        if fid == last_streamed_fid[0]:
            time.sleep(0.005)
            continue

        last_streamed_fid[0] = fid

        # Use overlay if it's recent (within 10 frames), else raw
        overlay_fresh = (fid - last_overlay_fid[0]) <= 10
        jpeg = last_overlay[0] if (last_overlay[0] is not None and overlay_fresh) else raw

        time.sleep(MIN_INTERVAL)

        # [FREEZE FIX] Yield frame, then pace with sleep.
        # Old code: yield then sleep — next frame ready only AFTER sleep finishes.
        # New code: keep MIN_INTERVAL sleep before checking new frame so CPU
        # is freed between yields without adding extra latency after yield.
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + jpeg + b'\r\n')

    try: stream_proc.kill()
    except: pass


# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────

# Camera-2 (OCR) URL — set by /api/start_camera, read by /api/video_feed_cam2
_camera2_url = None

# Camera-2 OCR engine instance (parallel thread — no Camera-1 impact)
camera2_ocr_instance = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    global _camera2_url
    data        = request.json or {}
    rtsp_url    = data.get('rtsp_url', '').strip()
    device_id   = data.get('device_id', 0)
    camera2_url = data.get('camera2_url', '').strip()   # OCR camera (UI only)

    state.stream_generation += 1
    state.camera_active      = False
    time.sleep(0.15)

    if rtsp_url:
        state.video_source    = rtsp_url
        msg = f'LAN Camera — {rtsp_url[:40]} — {state.backend.upper()}'
        print(f'[START_CAMERA] STREAM SOURCE set to {rtsp_url}')
    else:
        state.video_source    = None
        state.local_camera_id = int(device_id) if str(device_id).isdigit() else 0
        msg = f'Local Camera {state.local_camera_id} — {state.backend.upper()}'
        print(f'[START_CAMERA] Local camera selected: {state.local_camera_id}')

    # Store Camera-2 URL (OCR overlay) — no inference done here
    _camera2_url = camera2_url if camera2_url else None

    # ── Camera-2 OCR instance (ADD ONLY) ─────────────────────────────
    global camera2_ocr_instance
    if _CAM2_OCR_AVAILABLE and camera2_url:
        # Stop previous instance cleanly before creating a new one
        if camera2_ocr_instance is not None:
            try:
                camera2_ocr_instance.stop()
            except Exception:
                pass
        def _on_cam2_serial(serial):
            print(f'[CAM2-OCR] Callback → serial={serial}')
            if serial and serial not in (None, 'UNKNOWN', 'Searching...', 'Reading...'):
                # FIX A2: panel_id is None means reset_panel() already ran.
                # This is a late callback for the OLD panel — do NOT create a
                # new folder for the NEXT panel before its first image exists.
                if state.panel_id is None:
                    print(f'[CAM2-OCR] Late serial ignored (panel already reset): {serial}')
                    return
                state.serial_number = serial
                state.ocr_done      = True
                folder = state.current_sequence_panel_folder
                if folder:
                    threading.Thread(target=rename_panel_images, args=(folder, serial), daemon=True).start()

        # ── Resolve SSP_TAILGATE_SERIAL.pt path absolutely so it works regardless
        # of the process CWD (Flask can be launched from any directory).
        _script_dir       = os.path.dirname(os.path.abspath(__file__))
        _serial_pt_path   = os.path.join(_script_dir, "models", "SSP_TAILGATE_SERIAL.pt")
        if not os.path.exists(_serial_pt_path):
            # Also try the app root models folder if the script is started
            # from a different working directory.
            _serial_pt_path = os.path.join(SCRIPT_DIR, "models", "SSP_TAILGATE_SERIAL.pt")
        _pt_exists = os.path.exists(_serial_pt_path)
        print(f'[CAM2-OCR] SSP_TAILGATE_SERIAL.pt path  → {_serial_pt_path}')
        print(f'[CAM2-OCR] SSP_TAILGATE_SERIAL.pt found → {_pt_exists}')
        if not _pt_exists:
            print(f'[CAM2-OCR] ❌ CRITICAL: SSP_TAILGATE_SERIAL.pt missing — Camera 2 OCR will NOT work')
            print(f'[CAM2-OCR] ❌ Copy SSP_TAILGATE_SERIAL.pt to: {os.path.join(SCRIPT_DIR, "models", "SSP_TAILGATE_SERIAL.pt")}')
        else:
            import os as _os
            sz_mb = round(_os.path.getsize(_serial_pt_path) / (1024*1024), 1)
            print(f'[CAM2-OCR] ✅ SSP_TAILGATE_SERIAL.pt exists ({sz_mb} MB) — will load on camera start')

        # Open Camera 2 at highest practical resolution so serial digits are large.
        # 2048×1536 gives ~2x more pixels than 1280×960 for the serial region.
        camera2_ocr_instance = Camera2OCR(
            camera2_url=camera2_url,
            open_cap_fn=lambda u: _open_rtsp(u, width=2048, height=1536),
            on_serial_detected=_on_cam2_serial,
            pt_path=_serial_pt_path
        )
        print(f'[CAM2-OCR] Instance ready for: {camera2_url[:60]}')
    else:
        camera2_ocr_instance = None
        if not _CAM2_OCR_AVAILABLE:
            print('[CAM2-OCR] camera2_ocr.py not available — SSP_TAILGATE_SERIAL.pt/EasyOCR disabled')
        elif not camera2_url:
            print('[CAM2-OCR] No Camera-2 URL configured — SSP_TAILGATE_SERIAL.pt/EasyOCR disabled (set camera2_url)')

    state.camera_active = True
    return jsonify({
        'success':     True,
        'backend':     state.backend,
        'source':      'rtsp' if rtsp_url else 'camera',
        'message':     msg,
        'stream_gen':  state.stream_generation,
        'cam2_active': bool(_camera2_url),
    })


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """
    Accept a video upload and stream it straight to disk in 1 MB chunks.
    This works for any file size regardless of available RAM.
    """
    upload_dir = os.path.join(BASE_STORAGE, '_uploads')
    os.makedirs(upload_dir, exist_ok=True)

    # ── Get filename from Content-Disposition or fallback ─────
    cd          = request.headers.get('Content-Disposition', '')
    fname       = 'upload.mp4'
    for part in cd.split(';'):
        part = part.strip()
        if part.startswith('filename='):
            fname = part.split('=', 1)[1].strip(' "\'')
            break

    # Use form-data filename if available (multipart upload)
    if 'video' in request.files:
        f         = request.files['video']
        safe_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + \
                    os.path.basename(f.filename or fname)
        save_path = os.path.join(upload_dir, safe_name)
        # Stream to disk in 1 MB chunks
        CHUNK = 1024 * 1024
        with open(save_path, 'wb') as out:
            while True:
                chunk = f.stream.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)

    elif request.content_length or request.stream:
        # Raw body stream (application/octet-stream)
        safe_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + fname
        save_path = os.path.join(upload_dir, safe_name)
        CHUNK     = 1024 * 1024
        with open(save_path, 'wb') as out:
            while True:
                chunk = request.stream.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    else:
        return jsonify({'success': False, 'error': 'No video data received'}), 400

    # Verify file was saved and is readable
    if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
        return jsonify({'success': False, 'error': 'File save failed'}), 500

    file_mb = os.path.getsize(save_path) / (1024 * 1024)

    # Probe FPS and frame count
    cap          = cv2.VideoCapture(save_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Stop old main stream only if this upload is for Camera-1 playback.
    target = (request.form.get('target') or request.args.get('target') or 'cam1')
    if target == 'cam1':
        state.camera_active      = False
        time.sleep(0.1)
        state.stream_generation += 1
        state.video_source        = save_path
        state.camera_active       = True
        state.prev_gray           = None
        state.hand_bbox           = None
        state.reset_for_new_panel()
        print(f"[INFO] Video saved for Camera-1 playback: {safe_name} | {file_mb:.1f} MB | "
              f"{fps:.1f} FPS | {total_frames} frames")
    else:
        # Camera-2 uploads should not replace the main Camera-1 stream.
        print(f"[INFO] Video saved for Camera-2 OCR only: {safe_name} | {file_mb:.1f} MB | "
              f"{fps:.1f} FPS | {total_frames} frames")

    # Make uploaded Camera-2 files visible to the OCR overlay stream.
    if target == 'cam2':
        global _camera2_url
        _camera2_url = save_path

    # If target == cam2, start Camera-2 OCR instance using the uploaded file
    cam2_started = False
    if target == 'cam2':
        if _CAM2_OCR_AVAILABLE:
            global camera2_ocr_instance
            # Stop previous instance cleanly before creating a new one
            if camera2_ocr_instance is not None:
                try:
                    camera2_ocr_instance.stop()
                except Exception:
                    pass

            # Resolve SSP_TAILGATE_SERIAL.pt path (same logic as /api/start_camera)
            _script_dir       = os.path.dirname(os.path.abspath(__file__))
            _serial_pt_path   = os.path.join(_script_dir, "models", "SSP_TAILGATE_SERIAL.pt")
            if not os.path.exists(_serial_pt_path):
                _serial_pt_path = os.path.join(SCRIPT_DIR, "models", "SSP_TAILGATE_SERIAL.pt")
            print(f'[UPLOAD] camera2 upload -> SSP_TAILGATE_SERIAL.pt path {_serial_pt_path} exists={os.path.exists(_serial_pt_path)}')

            try:
                # Use a simple open_cap_fn that opens the saved file via OpenCV
                camera2_ocr_instance = Camera2OCR(
                    camera2_url=save_path,
                    open_cap_fn=lambda u, p=save_path: cv2.VideoCapture(p),
                    on_serial_detected=None,
                    pt_path=_serial_pt_path
                )
                # Start scanning immediately for uploaded video
                try:
                    state.serial_number = None
                    state.ocr_done = False
                    camera2_ocr_instance.reset_for_new_panel()
                    camera2_ocr_instance.start_ocr()
                except Exception:
                    pass
                cam2_started = True
                print(f'[UPLOAD] Camera-2 OCR started for uploaded file {safe_name}')
            except Exception as e:
                print(f'[UPLOAD] Failed to start Camera-2 OCR: {e}')
        else:
            print('[UPLOAD] Camera-2 OCR not available (camera2_ocr.py missing)')

    return jsonify({
        'success'     : True,
        'source'      : 'video',
        'target'      : target,
        'backend'     : state.backend,
        'fps'         : round(fps, 2),
        'total_frames': total_frames,
        'file_mb'     : round(file_mb, 1),
        'filename'    : safe_name,
        'target'      : target,
        'cam2_started': cam2_started,
        'message'     : f'Video ready — {fps:.1f} FPS real-time playback'
    })


@app.route('/api/list_cameras', methods=['GET'])
def list_cameras():
    """
    Probe cv2.VideoCapture indices 0–4 to find cameras physically connected to the Pi.
    Returns a list like: [{"index": 0, "label": "Camera 0 — 1280×720"}]
    This is completely different from browser mediaDevices which lists the PC's cameras.
    """
    found = []
    for idx in range(5):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    found.append({
                        'index': idx,
                        'label': f'Camera {idx}  —  {w}×{h}'
                    })
            cap.release()
        except Exception:
            pass
    if not found:
        found = [{'index': 0, 'label': 'Camera 0 (default)'}]
    print(f"[INFO] Cameras on Pi: {found}")
    return jsonify({'cameras': found})


@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """Test camera connection — waits for first frame from TLS stream."""
    data     = request.json or {}
    rtsp_url = data.get('rtsp_url', '').strip()
    if not rtsp_url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    try:
        cap = _open_rtsp(rtsp_url)
        if not cap.isOpened():
            return jsonify({'success': False,
                            'error': 'Cannot connect — check IP, port, credentials'})

        # Wait up to 12s for first frame (TLS stream needs time to decode first keyframe)
        import time as _time
        ok = False; frame = None
        for _ in range(120):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            _time.sleep(0.1)

        cap.release()
        if not ok or frame is None:
            return jsonify({'success': False,
                            'error': 'Connected but no video frame received — camera may be off or stream path wrong'})
        fh, fw = frame.shape[:2]
        return jsonify({
            'success': True,
            'width':   fw,
            'height':  fh,
            'fps':     25.0,
            'message': f'Connected — {fw}×{fh} @ 25 FPS'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    state.camera_active = False
    state.video_source  = None
    return jsonify({'success': True})


@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _generate_cam2_frames():
    """
    Camera-2 UI stream.
    Pulls frames from the OCR engine's _cap directly — always fresh.
    Auto-recovers if frame goes stale.
    """
    last_frame_ts    = [time.time()]
    last_frame_bytes = [None]
    STREAM_WATCHDOG  = 5.0
    STREAM_FPS       = 8

    def _make_banner(text, color_bgr=(255,255,255)):
        blank = np.full((960, 1280, 3), 20, dtype=np.uint8)
        cv2.rectangle(blank, (0, 0), (1280, 960), color_bgr, 8)
        cv2.putText(blank, text, (60, 480),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, color_bgr, 3)
        _, b = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return b.tobytes()

    while True:
        if not camera2_ocr_instance:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + _make_banner("CAMERA 2 NOT CONFIGURED") + b'\r\n')
            time.sleep(0.5)
            continue

        try:
            # Use get_annotated_frame() so the SSP_TAILGATE_SERIAL.pt bounding box is
            # always drawn on the Camera-2 stream — operators see the green
            # "serial" bbox the moment the model detects the serial region.
            frame = camera2_ocr_instance.get_annotated_frame()

            if frame is not None:
                last_frame_ts[0] = time.time()
                camera2_ocr_instance._last_ui_poll_ts = time.time()

                # ── Always rebuild the display frame on every loop iteration ──
                # The old hash-based check (`if fhash != last_frame_hash[0]:`)
                # prevented updates when the camera scene was static.
                # Consequence: the SSP_TAILGATE_SERIAL.pt bbox AND the OCR status text
                # (Scanning / Serial Captured) were NEVER updated once the
                # hash stabilised — so detection annotations were invisible
                # on any non-moving scene.
                # Fix: always encode and serve the latest annotated frame.
                display   = frame.copy()
                is_done   = camera2_ocr_instance.is_done()
                scanning  = getattr(camera2_ocr_instance, '_is_scanning', False)
                serial    = camera2_ocr_instance.get_serial_number()
                status    = getattr(camera2_ocr_instance, 'status', '')
                yolo_det   = camera2_ocr_instance.get_latest_detection()
                yolo_ready = camera2_ocr_instance.serial_detector is not None

                # ── SSP_TAILGATE_SERIAL.pt status overlay ──────────────────────────────
                # Draw a prominent red banner when SSP_TAILGATE_SERIAL.pt is not loaded so
                # the operator immediately knows annotations are disabled.
                if not yolo_ready:
                    h_f, w_f = display.shape[:2]
                    # Check whether file exists to give a precise message
                    _pt_chk = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "models", "SSP_TAILGATE_SERIAL.pt")
                    _loading = getattr(camera2_ocr_instance, '_yolo_loading', False)
                    if _loading:
                        _msg = "SSP_TAILGATE_SERIAL.pt LOADING... please wait"
                        _clr = (0, 140, 255)  # orange
                    elif not os.path.exists(_pt_chk):
                        _msg = f"SSP_TAILGATE_SERIAL.pt MISSING — copy to models/SSP_TAILGATE_SERIAL.pt"
                        _clr = (0, 0, 220)    # red
                    else:
                        _msg = "SSP_TAILGATE_SERIAL.pt LOAD FAILED — check terminal for error"
                        _clr = (0, 0, 200)    # dark red
                    # Draw 2-line banner at bottom
                    cv2.rectangle(display, (0, h_f - 60), (w_f, h_f), _clr, -1)
                    cv2.putText(display, _msg,
                                (8, h_f - 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display, "NO SERIAL ANNOTATION — OCR DISABLED",
                                (8, h_f - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (200, 200, 255), 1, cv2.LINE_AA)

                # ── Status text — top-left corner ─────────────────────────
                if is_done and serial and serial != "Reading...":
                    status_txt = f"Serial: {serial}  ✓"
                    txt_color  = (0, 220, 80)    # green
                elif scanning and yolo_det is not None:
                    status_txt = "Scanning — serial detected"
                    txt_color  = (0, 220, 255)   # yellow
                elif scanning:
                    status_txt = status or "Scanning..."
                    txt_color  = (0, 180, 255)   # amber
                else:
                    status_txt = ""
                    txt_color  = (180, 180, 180)

                if status_txt:
                    (tw, th), _ = cv2.getTextSize(
                        status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    pad = 8
                    cv2.rectangle(display,
                                  (10, 8),
                                  (10 + tw + pad*2, 8 + th + pad*2),
                                  (20, 20, 20), -1)
                    cv2.putText(display, status_txt,
                                (10 + pad, 8 + th + pad),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                txt_color, 2, cv2.LINE_AA)

                _, jpg = cv2.imencode('.jpg', display,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
                last_frame_bytes[0] = jpg.tobytes()

                # Serve latest frame (fresh or cached)
                age = time.time() - last_frame_ts[0]
                if age > STREAM_WATCHDOG:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + _make_banner("CAMERA 2 RECONNECTING...",
                                          (0, 80, 200)) + b'\r\n')
                    time.sleep(0.5)
                    continue

                if last_frame_bytes[0]:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + last_frame_bytes[0] + b'\r\n')
            else:
                age = time.time() - last_frame_ts[0]
                if age > STREAM_WATCHDOG:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + _make_banner("CAMERA 2 CONNECTING...",
                                          (0, 100, 255)) + b'\r\n')
                time.sleep(0.3)
                continue

        except Exception as e:
            print(f'[CAM2 STREAM] {e}')
            time.sleep(0.5)
            continue

        time.sleep(1.0 / STREAM_FPS)



@app.route('/api/video_feed_cam2')
def video_feed_cam2():
    """Camera-2 OCR overlay stream — raw MJPEG passthrough, no inference."""
    return Response(_generate_cam2_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cam2_status')
def cam2_status():
    """Camera-2 OCR state — polled by UI every 500ms for yellow/green/red."""
    if not _CAM2_OCR_AVAILABLE or camera2_ocr_instance is None:
        return jsonify({'available': False, 'state': 'offline',
                        'serial': None, 'partial': None, 'attempts': 0,
                        'ocr_status': 'Camera 2 offline'})
    done       = camera2_ocr_instance.is_done()
    serial     = camera2_ocr_instance.get_serial_number()
    partial    = getattr(camera2_ocr_instance, 'partial_serial', None)
    scanning   = getattr(camera2_ocr_instance, '_is_scanning', False)
    attempts   = getattr(camera2_ocr_instance, 'ocr_attempt_count', 0)
    ocr_status = getattr(camera2_ocr_instance, 'status', '')

    # Phase-2 in progress: frame captured but serial not yet read
    phase2_reading = done and serial == "Reading..."

    # Update main state serial only once Phase-2 produces a real serial
    if done and serial and serial != "Reading..." \
            and (state.serial_number in (None, 'UNKNOWN', 'Searching...', 'Reading...')):
        state.serial_number = serial
        state.ocr_done      = True
        print(f"[CAM2] Serial updated from Phase-2: {serial}")
        
        # Rename existing images in the folder to include the serial
        folder = ensure_panel_folder()
        rename_panel_images(folder, serial)

    # Sync partial serial to state for sequence_status API
    if partial and partial not in ('Reading...', 'UNKNOWN'):
        state.partial_serial = partial

    # Determine the state key the UI understands
    if done and not phase2_reading:
        ui_state = 'found'
    elif done and phase2_reading:
        ui_state = 'captured'   # green immediately, serial still being read
    elif scanning:
        ui_state = 'scanning'
    else:
        ui_state = 'idle'

    _yolo_loaded  = camera2_ocr_instance.serial_detector is not None
    _yolo_loading = getattr(camera2_ocr_instance, '_yolo_loading', False)
    _frames       = getattr(camera2_ocr_instance, '_frame_count', 0)
    _dets         = getattr(camera2_ocr_instance, '_yolo_det_count', 0)

    # Give the UI a precise SSP_TAILGATE_SERIAL.pt status string
    if _yolo_loaded:
        _pt_status = f"✅ loaded ({camera2_ocr_instance.serial_detector._device})"
    elif _yolo_loading:
        _pt_status = "⏳ loading..."
    else:
        _pt_chk = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "SSP_TAILGATE_SERIAL.pt")
        _pt_status = ("❌ file missing — copy SSP_TAILGATE_SERIAL.pt to models/" 
                      if not os.path.exists(_pt_chk) 
                      else "❌ load failed — check terminal")

    return jsonify({
        'available':    True,
        'state':        ui_state,
        'serial':       serial,
        'partial':      partial,
        'attempts':     attempts,
        'ocr_status':   ocr_status,
        'serial_pt':    _pt_status,
        'yolo_loaded':  _yolo_loaded,
        'frames_read':  _frames,
        'detections':   _dets,
    })


@app.route('/api/sequence_status', methods=['GET'])
def sequence_status():
    # Only refresh date (not directory count) — count is maintained in memory
    new_date = datetime.now().strftime("%Y-%m-%d")
    if new_date != state.today_date:
        state.today_date        = new_date
        state.panel_count_today = 0   # midnight reset
        # Persist reset
        try:
            count_file = os.path.join(BASE_STORAGE, 'panel_count.txt')
            with open(count_file, 'w') as _f:
                _f.write(f"{new_date},0")
        except Exception:
            pass

    # 5. NORMALIZE SEQUENCE STATUS (FIX UI BUG)
    model_sees_seq = getattr(state, 'model_sees', 'NONE')
    for i in [1, 2, 3]:
        if getattr(state, 'completed', {}).get(i):
            # ── Sequence is done — keep whatever final status was set ──
            # completed[i]=True is set by complete() which sets 'completed',
            # AND also by the missed path which sets 'missed'. So read the
            # real sequence_status[i] rather than forcing 'completed'.
            # If it hasn't been set to 'missed' explicitly, default to 'completed'.
            if state.sequence_status.get(i) not in ('completed', 'missed'):
                state.sequence_status[i] = 'completed'
            # (else: keep existing 'completed' or 'missed' — do not overwrite)
        elif i == 3 and state.sequence_captured.get(3, False) and state.all_sequences_done:
            # SEQ3 image captured and all_sequences_done → mark completed in UI.
            if state.sequence_status.get(3) not in ('completed', 'missed'):
                state.sequence_status[3] = 'completed'
                state.completed[3] = True
                print("[STATUS] SEQ3 auto-completed via sequence_captured flag")
        elif i == state.current_sequence:
            # Use BOTH wiping_active flag AND status_msg check so the UI badge
            # never lags a poll cycle behind the actual wipe state.
            is_wiping = (state.wiping_active or
                         "WIPING" in getattr(state, 'status_msg', ""))
            state.sequence_status[i] = "wiping" if is_wiping else "active"
        elif state.sequence_status.get(i) == 'missed':
            # Preserve 'missed' — don't let poll cycle overwrite it to 'pending'
            pass
        elif (state.current_sequence == 0
              and not state.all_sequences_done
              and not state.completed.get(i, False)
              and model_sees_seq == f"panel_seq{i}"
              and getattr(state, 'panel_absent_frames', 99) < 5):
            # IDLE + model actively sees panel live — give operator live feedback.
            # FIX B3: guards prevent ghost badge after panel removal.
            state.sequence_status[i] = "active"
        else:
            state.sequence_status[i] = "pending"

    # Build per-sequence cleaning percentage
    per_seq_pct = {}
    for i in (1, 2, 3):
        if state.sequence_status.get(i) == 'completed':
            per_seq_pct[str(i)] = 100.0
        elif i == state.current_sequence:
            per_seq_pct[str(i)] = state.progress
        else:
            per_seq_pct[str(i)] = 0.0

    # Convert sequence_status to string keys for JSON compatibility
    seq_status_str = {str(k): v for k, v in state.sequence_status.items()}

    # ── FIX TOTAL LEAD TIME ───────────────────────
    total_time = 0
    if getattr(state, 'panel_end_time', None) and getattr(state, 'panel_start_time', None):
        total_time = state.panel_end_time - state.panel_start_time
    elif getattr(state, 'panel_start_time', None):
        total_time = time.time() - state.panel_start_time

    seq_times = {
        1: (state.seq_end_time.get(1) or time.time()) - (state.seq_start_time.get(1) or time.time()) if state.seq_start_time.get(1) else 0,
        2: (state.seq_end_time.get(2) or time.time()) - (state.seq_start_time.get(2) or time.time()) if state.seq_start_time.get(2) else 0,
        3: (state.seq_end_time.get(3) or time.time()) - (state.seq_start_time.get(3) or time.time()) if state.seq_start_time.get(3) else 0,
    }

    # ── Landscape alert lifecycle ─────────────────────────────────
    # SEQ2/SEQ3: When image is captured, mark it visually but NO voice announcement
    # (User requirement: remove 'seq completed' voice)
    if (state.landscape_alert == "seq2"
            and state.seq2_auto_captured
            and not state._seq2_cap_announced):
        state._seq2_cap_announced = True
        state.landscape_alert     = "captured"
        state.landscape_alert_ts  = time.time()
        # Voice removed per user request — no "captured" announcement

    elif (state.landscape_alert == "seq3"
            and state.seq3_auto_captured
            and not state._seq3_cap_announced):
        state._seq3_cap_announced = True
        state.landscape_alert     = "captured"
        state.landscape_alert_ts  = time.time()
        # Voice removed per user request — no "captured" announcement

    # Auto-clear "captured" green banner after 3 seconds
    elif (state.landscape_alert == "captured"
            and time.time() - state.landscape_alert_ts > 3.0):
        state.landscape_alert = ""

    return jsonify({
        "sequence": state.current_sequence,
        "status": seq_status_str,
        # Return '' for UI until a real serial is confirmed — resets blank on new panel
        "serial": state.serial_number
                  if state.serial_number not in (None, 'UNKNOWN', 'Searching...', 'Reading...', '')
                  else '',
        "reset_id": state.panel_reset_id,
        'seq_time': {str(k): v for k, v in seq_times.items()},
        'total_time': total_time,

        # Keep existing fields so nothing breaks
        'current_sequence':    state.current_sequence,
        'sequence_status':     seq_status_str,
        'seq_phase':           {str(k): v for k, v in state.seq_phase.items()},
        'cleaning_percentage': state.progress,
        'per_seq_pct':         per_seq_pct,
        # Blank for UI header until confirmed; '' resets UI on new panel
        'serial_number':       state.serial_number
                               if state.serial_number not in (None, 'UNKNOWN', 'Searching...', 'Reading...', '')
                               else '',
        # ADDED: UI needs this to know when to show green serial number
        'ocr_confirmed':       (state.ocr_done and state.serial_number 
                                and state.serial_number not in (None, 'UNKNOWN', 'Searching...', 'Reading...', '')),
        'captures':            {str(k): v for k, v in state.sequence_captured.items()},
        'backend':             state.backend,
        'panel_count_today':   state.panel_count_today,
        'today_date':          state.today_date,
        # Show partial only while scanning; hide once full serial confirmed
        'partial_serial':      (state.partial_serial
                                if not state.ocr_done
                                else None)
                               or (getattr(camera2_ocr_instance, 'partial_serial', None)
                                   if camera2_ocr_instance and not state.ocr_done
                                   else None),
        'wiping_active':       state.wiping_active,
        'current_panel_name':  getattr(state, 'current_panel_name', state.current_sequence_panel_name),
        'all_sequences_done':  state.all_sequences_done,
        'panel_reset_id':      state.panel_reset_id,
        'failure_reason':      getattr(state, 'failure_reason', None),
        # ── Landscape placement alert (voice + UI colour) ──────────
        # "" = nothing | "seq2" = orange | "seq3" = blue | "captured" = green
        'landscape_alert':     getattr(state, 'landscape_alert', ''),
        # Camera-2 fields — needed by UI auto-reconnect on page refresh
        'cam2_active':         bool(_camera2_url),
        'cam2_url':            _camera2_url or '',
        'lead_times': {
            str(i): round(
                (state.seq_end_time.get(i) or time.time()) -
                (state.seq_start_time.get(i) or time.time()), 1)
            if state.seq_start_time.get(i) else None
            for i in (1, 2, 3)
        },
        'total_lead_time': round(
            ((state.panel_end_time or time.time()) -
             (state.panel_start_time or time.time())), 1)
        if state.panel_start_time else None,
    })


@app.route('/api/serial-status', methods=['GET'])
def serial_status():
    """
    Check SSP_TAILGATE_SERIAL.pt loading and Camera 2 OCR status.
    Call from browser: GET /api/serial-status
    Shows exactly why SSP_TAILGATE_SERIAL.pt may not be loaded.
    """
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _pt_path    = os.path.join(_script_dir, "models", "SSP_TAILGATE_SERIAL.pt")
    pt_exists   = os.path.exists(_pt_path)
    pt_size_mb  = round(os.path.getsize(_pt_path) / (1024*1024), 1) if pt_exists else 0

    diag = {
        'serial_pt_path':    _pt_path,
        'serial_pt_exists':  pt_exists,
        'serial_pt_size_mb': pt_size_mb,
        'cam2_instance':     camera2_ocr_instance is not None,
    }

    if camera2_ocr_instance is not None:
        diag['yolo_loaded']      = camera2_ocr_instance.serial_detector is not None
        diag['yolo_loading']     = getattr(camera2_ocr_instance, '_yolo_loading', False)
        diag['yolo_device']      = (str(getattr(camera2_ocr_instance.serial_detector, '_device', ''))
                                    if camera2_ocr_instance.serial_detector else None)
        diag['frames_received']  = getattr(camera2_ocr_instance, '_frame_count', 0)
        diag['detections_total'] = getattr(camera2_ocr_instance, '_yolo_det_count', 0)
        diag['ocr_scanning']     = camera2_ocr_instance._is_scanning
        diag['ocr_done']         = camera2_ocr_instance.ocr_done
        diag['serial_number']    = camera2_ocr_instance.serial_number

        if not pt_exists:
            diag['status'] = '❌ SERIAL.PT FILE NOT FOUND — place it in models/ folder'
        elif not diag['yolo_loaded'] and diag['yolo_loading']:
            diag['status'] = '⏳ SSP_TAILGATE_SERIAL.pt loading (wait a few seconds)'
        elif not diag['yolo_loaded']:
            diag['status'] = '❌ SSP_TAILGATE_SERIAL.pt failed to load — check terminal for error'
        else:
            diag['status'] = f"✅ SSP_TAILGATE_SERIAL.pt loaded on {diag['yolo_device']}"
    else:
        diag['status'] = '❌ Camera 2 not started — enter Camera 2 URL and start camera'

    return jsonify(diag)


@app.route('/api/serial-number-result', methods=['GET'])
def serial_number_result():
    """
    Poll serial number detection result with all detected digits.
    Shows what was detected, candidates, confidence, and frame info.
    
    Returns:
    {
        'status': 'scanning' | 'detected' | 'confirmed' | 'error',
        'serial_number': '2024-AB12' (if confirmed),
        'confidence': 0.94,
        'digits': {
            'day': {
                'value': '20',
                'candidates': ['20', '28', '30'],
                'confidence': 0.98
            },
            'code': {
                'value': 'AB',
                'candidates': ['AB', 'AO', 'A8'],
                'confidence': 0.94
            },
            'letter': {
                'value': '12',
                'candidates': ['12', '13', '14'],
                'confidence': 0.90
            }
        },
        'frames_saved': {
            'best_frame': 'CAM2_BEST_SERIAL_FRAME.png',
            'location': '/panel_data/YYYY-MM-DD/SSP-SEQ_HHMMSS/',
            'resolution': '1280x960',
            'format': 'PNG (lossless)'
        }
    }
    """
    result = {
        'status': 'idle',
        'serial_number': None,
        'confidence': 0.0,
        'digits': {
            'day': {'value': None, 'candidates': [], 'confidence': 0.0},
            'code': {'value': None, 'candidates': [], 'confidence': 0.0},
            'letter': {'value': None, 'candidates': [], 'confidence': 0.0}
        },
        'frames_saved': {
            'best_frame': None,
            'location': None,
            'resolution': '1280x960',
            'format': 'PNG (lossless)'
        },
        'message': 'No serial detection yet'
    }
    
    if camera2_ocr_instance is None:
        result['status'] = 'error'
        result['message'] = '❌ Camera 2 not initialized'
        return jsonify(result)
    
    # Get current serial detection state
    if camera2_ocr_instance._is_scanning:
        result['status'] = 'scanning'
        result['message'] = '🔍 Scanning for serial number...'
    elif camera2_ocr_instance.ocr_done:
        result['status'] = 'confirmed'
        result['serial_number'] = camera2_ocr_instance.serial_number
        result['confidence'] = getattr(camera2_ocr_instance, 'serial_confidence', 0.0)
        result['message'] = f"✅ Serial confirmed: {camera2_ocr_instance.serial_number}"
    elif camera2_ocr_instance.serial_number:
        result['status'] = 'detected'
        result['serial_number'] = camera2_ocr_instance.serial_number
        result['confidence'] = getattr(camera2_ocr_instance, 'serial_confidence', 0.0)
        result['message'] = f"🔍 Serial detected: {camera2_ocr_instance.serial_number} (waiting for confirmation)"
    else:
        result['status'] = 'scanning'
        result['message'] = '🔍 Searching for serial number...'
    
    # Get digit candidates if available
    digit_candidates = getattr(camera2_ocr_instance, 'digit_candidates', {})
    if digit_candidates:
        for key in ['day', 'code', 'letter']:
            if key in digit_candidates and digit_candidates[key]:
                candidates_data = digit_candidates[key]
                if isinstance(candidates_data, list) and len(candidates_data) > 0:
                    result['digits'][key]['candidates'] = [c[0] if isinstance(c, (list, tuple)) else c for c in candidates_data]
                    if isinstance(candidates_data[0], (list, tuple)) and len(candidates_data[0]) > 1:
                        result['digits'][key]['confidence'] = candidates_data[0][1]
                        result['digits'][key]['value'] = candidates_data[0][0]
    
    # Get best frame location if saved
    best_frame_path = getattr(camera2_ocr_instance, '_best_serial_frame_path', None)
    if best_frame_path and os.path.exists(best_frame_path):
        result['frames_saved']['best_frame'] = os.path.basename(best_frame_path)
        result['frames_saved']['location'] = os.path.dirname(best_frame_path)
    
    return jsonify(result)


@app.route('/api/reset', methods=['POST'])
def reset_system():
    state.reset_for_new_panel()
    return jsonify({'success': True, 'message': 'System reset'})


@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """
    Comprehensive diagnostic check for SSP_TAILGATE_SERIAL.pt, Camera-2, and OCR status.
    Verifies that all components are loaded and working correctly.
    """
    diag = {}
    
    # 1. Check SSP_TAILGATE_SERIAL.pt file
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _serial_pt_path = os.path.join(_script_dir, "models", "SSP_TAILGATE_SERIAL.pt")
    if not os.path.exists(_serial_pt_path):
        _serial_pt_path = os.path.join(SCRIPT_DIR, "models", "SSP_TAILGATE_SERIAL.pt")
    
    diag['serial_pt_path'] = _serial_pt_path
    diag['serial_pt_exists'] = os.path.exists(_serial_pt_path)
    if os.path.exists(_serial_pt_path):
        diag['serial_pt_size_mb'] = round(os.path.getsize(_serial_pt_path) / (1024*1024), 1)
    
    # 2. Check Camera2OCR instance
    diag['camera2_ocr_available'] = _CAM2_OCR_AVAILABLE
    diag['camera2_ocr_instance_exists'] = camera2_ocr_instance is not None
    
    if camera2_ocr_instance is not None:
        # 3. Check YOLO detector
        diag['yolo_detector_loaded'] = camera2_ocr_instance.serial_detector is not None
        if camera2_ocr_instance.serial_detector is not None:
            diag['yolo_device'] = camera2_ocr_instance.serial_detector._device
            diag['yolo_status'] = "✅ Ready"
        else:
            diag['yolo_status'] = f"⚠️ Loading... (still initializing)" if getattr(camera2_ocr_instance, '_yolo_loading', True) else "❌ Failed to load"
        
        # 4. Check EasyOCR
        diag['easyocr_loaded'] = camera2_ocr_instance.easy_reader is not None
        diag['easyocr_status'] = "✅ Ready" if camera2_ocr_instance.easy_reader is not None else "⚠️ Not loaded yet"
        
        # 5. Camera-2 frame reading status
        diag['camera2_url'] = getattr(camera2_ocr_instance, 'url', 'N/A')[:60] if getattr(camera2_ocr_instance, 'url', None) else "Not set"
        diag['last_frame_timestamp'] = camera2_ocr_instance._last_good_ts_global if hasattr(camera2_ocr_instance, '_last_good_ts_global') else None
        diag['frame_reading_active'] = camera2_ocr_instance.running
        
        # 6. OCR scanning status
        diag['ocr_scanning'] = camera2_ocr_instance.is_scanning
        diag['ocr_done'] = camera2_ocr_instance.ocr_done
        diag['current_serial'] = camera2_ocr_instance.serial_number or 'Not detected yet'
        
        # 7. Frame detection status
        diag['latest_detection'] = camera2_ocr_instance._latest_det is not None
        diag['detection_timestamp'] = camera2_ocr_instance._last_yolo_detect_ts if hasattr(camera2_ocr_instance, '_last_yolo_detect_ts') else 0
        
        # 8. Capture slots status
        slots_status = []
        for i, slot in enumerate(camera2_ocr_instance._capture_slots):
            slots_status.append({
                'slot': i+1,
                'filled': slot is not None,
                'saved': camera2_ocr_instance._slots_saved[i] if i < len(camera2_ocr_instance._slots_saved) else False
            })
        diag['capture_slots'] = slots_status
        
        # 9. Panel folder
        diag['panel_folder'] = camera2_ocr_instance.panel_folder
        diag['panel_folder_valid'] = os.path.isdir(camera2_ocr_instance.panel_folder)
        
        # 10. Overall status
        overall_status = "✅ Fully Operational"
        if not diag['yolo_detector_loaded']:
            overall_status = "⚠️ YOLO loading..."
        if not diag['easyocr_loaded']:
            overall_status = "⚠️ EasyOCR loading..."
        diag['overall_status'] = overall_status
    else:
        diag['overall_status'] = "❌ Camera2OCR not initialized"
    
    return jsonify(diag)


@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Shows raw model output to diagnose class name issues."""
    return jsonify({
        'current_sequence':     state.current_sequence,
        'sequence_status':      state.sequence_status,
        'cleaning_percentage':  state.progress,
        'current_panel_name':   getattr(state, 'current_sequence_panel_name', None),
        'serial_number':        state.serial_number,
        'detected_seq_history': list(state.detected_seq_history)[-10:],
        'wiping_active':        state.wiping_active,
        'no_wiping_frames':     state.no_wiping_frames,
        'hand_bbox':            state.hand_bbox,
        'panel_rect':           state.panel_rect,
        'all_sequences_done':   state.all_sequences_done,
        'panel_count_today':    state.panel_count_today,
    })


@app.route('/api/next_sequence', methods=['POST'])
def next_sequence_manual():
    ok = advance_sequence()
    return jsonify({'success': ok,
                    'current_sequence': state.current_sequence})


@app.route('/api/capture_now', methods=['POST'])
def capture_now():
    # FIX CRASH3: numpy arrays cannot be used with Python "or" —
    # use explicit is-None guards throughout.
    best_frame = state.last_clean_frame
    if best_frame is None: best_frame = state.orig_frame
    if best_frame is None:
        return jsonify({'success': False, 'error': 'No frame available'}), 400
    _pr = state.last_clean_panel_rect
    if _pr is None: _pr = state.panel_rect
    _pc = state.last_clean_contour
    if _pc is None: _pc = state.panel_contour
    cap_data = {
        'frame':         best_frame,
        'panel_rect':    _pr,
        'panel_contour': _pc,
        'serial_det':    state.last_clean_serial_det or state.serial_det,
    }
    saved = capture_sequence_images(cap_data, state.current_sequence, state.serial_number)
    if saved:
        state.sequence_captured[state.current_sequence] = True
        return jsonify({'success': True,
                        'sequence': state.current_sequence})
    return jsonify({'success': False, 'error': 'Capture failed'}), 500


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("[OK] PANEL VISION INSPECTION SYSTEM")
    print("    Vidana Consulting Pvt Ltd")
    print("="*60)
    be = state.backend.upper()
    print(f"[INFO] Backend  : {be}")
    print(f"[INFO] SCRIPT_DIR: {os.path.abspath(SCRIPT_DIR)}")
    print(f"[INFO] BASE_STORAGE: {os.path.abspath(BASE_STORAGE)}")
    print(f"[INFO] TEMP_DIR: {os.path.abspath(TEMP_DIR)}")
    if state.backend in ("cpu", "gpu"):
        print(f"[INFO] Model    : {PT_MODEL_PATH}")
        try:
            print(f"MODEL PATH = {os.path.abspath(PT_MODEL_PATH)}")
        except Exception:
            pass
    else:
        print("[WARN] No model — hand detection disabled")
    print(f"📁  Storage  : {os.path.abspath(BASE_STORAGE)}")
    # Print configured stream source (None until camera started)
    try:
        stream_src = state.video_source if getattr(state, 'video_source', None) else 'None'
        print(f"STREAM SOURCE = {stream_src}")
    except Exception:
        pass
    print(f"[INFO] FFMPEG  : {FFMPEG_BIN if FFMPEG_BIN else 'not found'}")
    print(f"🌐  Server   : http://localhost:5000")
    print("="*60 + "\n")
    
    # ── Initialize logging ────────────────────────────────────────────────────
    log.init(BASE_STORAGE)
    log.app_start(
        cam1_url=state.video_source or "Not configured yet",
        cam2_url=_camera2_url or "Not configured yet"
    )
    print("[LOGGING] Daily log initialized in panel_data/YYYY-MM-DD/")
    
    # ── AUTO-START CAMERAS FROM config.json ───────────────────────────────────
    config_file = os.path.join(SCRIPT_DIR, "config.json")
    if os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                cam1_url = config.get('camera1_url')
                cam2_url = config.get('camera2_url')
                SEQ1_NORMAL_FRAMES = config.get('seq1_normal_frames', SEQ1_NORMAL_FRAMES)
                SEQ1_FALLBACK_FRAMES = config.get('seq1_fallback_frames', SEQ1_FALLBACK_FRAMES)
                SEQ_WIPE_SECONDS = config.get('seq_wipe_seconds', SEQ_WIPE_SECONDS)
                SEQ_HAND_GONE_SECS = config.get('seq_hand_gone_secs', SEQ_HAND_GONE_SECS)
                _SEQ2_GRACE_PERIOD = config.get('seq2_grace_period', _SEQ2_GRACE_PERIOD)
                _SEQ3_GRACE_PERIOD = config.get('seq3_grace_period', _SEQ3_GRACE_PERIOD)
                _SEQ2_COUNTDOWN_DURATION = config.get('seq2_countdown_duration', _SEQ2_COUNTDOWN_DURATION)
                _SEQ3_COUNTDOWN_DURATION = config.get('seq3_countdown_duration', _SEQ3_COUNTDOWN_DURATION)
                # NEW: Load capture delay and progress bar duration from config
                CAPTURE_DELAY_SECS = config.get('capture_delay_secs', CAPTURE_DELAY_SECS)
                PROGRESS_BAR_DURATION_SECS = config.get('progress_bar_duration_secs', PROGRESS_BAR_DURATION_SECS)
                print(f"[CONFIG] capture_delay_secs={CAPTURE_DELAY_SECS}s  progress_bar_duration_secs={PROGRESS_BAR_DURATION_SECS}s")
            print(f"[CONFIG] Loaded: {config_file}")
            if cam1_url:
                print(f"[CONFIG] CAM1: {cam1_url[:50]}")
            if cam2_url:
                print(f"[CONFIG] CAM2: {cam2_url[:50]}")
            if cam1_url or cam2_url:
                print("[AUTO-START] Starting cameras in background...")
                def _auto_start_cameras():
                    try:
                        global _camera2_url, camera2_ocr_instance
                        if cam1_url:
                            state.video_source = cam1_url
                            state.stream_generation += 1
                            print(f"[AUTO-START] CAM1 configured")
                        if _CAM2_OCR_AVAILABLE and cam2_url:
                            _camera2_url = cam2_url
                            _serial_pt = os.path.join(SCRIPT_DIR, "models", "SSP_TAILGATE_SERIAL.pt")
                            def _on_serial(s):
                                if s and s not in (None, 'UNKNOWN', 'Searching...'):
                                    state.serial_number = s
                                    state.ocr_done = True
                            camera2_ocr_instance = Camera2OCR(
                                camera2_url=cam2_url,
                                open_cap_fn=lambda u: _open_rtsp(u, width=2048, height=1536),
                                on_serial_detected=_on_serial,
                                pt_path=_serial_pt
                            )
                            print(f"[AUTO-START] CAM2 initialized")
                        state.camera_active = True
                        print("[AUTO-START] Cameras ready ✅")
                    except Exception as e:
                        print(f"[AUTO-START] Error: {e}")
                        import traceback
                        traceback.print_exc()
                threading.Thread(target=_auto_start_cameras, daemon=True).start()
        except Exception as e:
            print(f"[CONFIG] Error reading config.json: {e}")
    else:
        print(f"[CONFIG] config.json not found at {config_file}")
        print(f"[CONFIG] Create it with: {{'camera1_url': 'rtsp://...', 'camera2_url': 'rtsp://...'}}")

    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)