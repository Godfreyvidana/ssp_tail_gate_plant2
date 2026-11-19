# -*- coding: utf-8 -*-
"""
ocr_service.py  —  SSP Panel Vision  |  Vidana Consulting Pvt Ltd
=================================================================
Standalone OCR Microservice  (port 5001)

Architecture
────────────
  Port 5000  app_vision.py   — RTSP + YOLO + Live Streams + Flask UI
  Port 5001  ocr_service.py  — PaddleOCR only  (THIS FILE)

PaddleOCR runs in its own Python process here.  It NEVER shares the
GIL with the main app, so serial extraction never freezes the video
feed.

Endpoints
─────────
  GET  /health        → {"status": "ready"} once PaddleOCR is loaded
                        {"status": "loading"} while still initialising
  POST /ocr           → body: multipart image OR JSON {"image_b64": "..."}
                        response: {"serial": "...", "raw_text": "...", "score": 0.98}
                                  or {"serial": null, "raw_text": "", "score": 0}
  GET  /status        → {"status": ..., "calls": N, "avg_ms": ...}

Usage
─────
  .venv\\Scripts\\python.exe ocr_service.py
  # or:
  run_all.bat
"""

from __future__ import annotations

import os
import re
import sys
import time
import base64
import string
import threading
import traceback
from collections import Counter
from datetime import datetime

# ── Limit CPU threads so OCR doesn't starve the network/event loop ──
os.environ.setdefault('OMP_NUM_THREADS',       '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS',  '2')
os.environ.setdefault('MKL_NUM_THREADS',       '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('NUMEXPR_NUM_THREADS',   '2')

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OCR_SERVICE_PORT = 5001

print("\n" + "="*62)
print(">>> [OCR-SERVICE] PaddleOCR Microservice  port=5001 <<<")
print("="*62 + "\n")

# ─────────────────────────────────────────────────────────────
# SERIAL NUMBER VALIDATION  (ported from camera2_ocr.py)
# ─────────────────────────────────────────────────────────────

_REPL_DIGITS = {
    '/':'7', '|':'1', 'I':'1', 'i':'1',
    ' ':'_', 'J':'1', ':':'_', '*':'_', ';':'_',
}
_REPL_CHAR = {'4':'A', '^':'A', 'a':'A', '8':'B'}


def _apply_corrections(s: str):
    """Exact port of reference validate_serial_strict() + correct_serial()."""
    if len(s) not in (6, 10):
        return None
    if len(s) == 6:
        day_s, letter_s, code_s = s[0:2], s[2], s[3:6]
    else:
        day_s, letter_s, code_s = s[0:2], s[6], s[7:10]

    def fix_digits(t):
        return (t.replace('l','1').replace('L','1')
                 .replace('i','1').replace('I','1')
                 .replace('O','0').replace('o','0')
                 .replace('/','7').replace('|','1')
                 .replace('A','4').replace('a','4'))

    day_s  = fix_digits(day_s.upper())
    code_s = fix_digits(code_s.upper())

    try:
        day_int = int(day_s)
        if not (1 <= day_int <= 31):
            return None
    except ValueError:
        return None

    if not (code_s.isdigit() and len(code_s) == 3):
        return None

    letter_s = letter_s.upper()
    if   letter_s == '4':            letter_s = 'A'
    elif letter_s in ('3', '8'):     letter_s = 'B'
    elif letter_s == '^':            letter_s = 'A'
    elif letter_s == 'O':            letter_s = 'A'

    if letter_s not in string.ascii_uppercase:
        return None

    _n = datetime.now()
    return day_s + f"{_n.month:02d}{_n.year%100:02d}" + letter_s + code_s


def _correct_serial(raw: str):
    """Clean OCR noise then extract valid 9-digit+1-letter serial."""
    s = raw.strip().strip("'`\"")
    for k, v in _REPL_DIGITS.items():
        s = s.replace(k, v)
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    if not (9 <= len(s) <= 12):
        return None
    if len(s) == 10:
        return _apply_corrections(s)
    if len(s) == 11:
        return _apply_corrections(s[1:]) or _apply_corrections(s[:-1])
    if len(s) == 12:
        return (_apply_corrections(s[2:]) or _apply_corrections(s[:-2]) or
                _apply_corrections(s[1:-1]))
    return None


# ─────────────────────────────────────────────────────────────
# PADDLEOCR ENGINE  (loads once in background at startup)
# ─────────────────────────────────────────────────────────────

class _OCREngine:
    """Singleton wrapper around PaddleOCR — thread-safe."""

    def __init__(self):
        self._reader   = None
        self._ready    = False
        self._failed   = False
        self._lock     = threading.Lock()
        self._calls    = 0
        self._total_ms = 0.0
        threading.Thread(target=self._init, daemon=True).start()

    def _init(self):
        print("[OCR-SERVICE] Initialising PaddleOCR (CPU) …", flush=True)
        try:
            from paddleocr import PaddleOCR
            reader = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                enable_mkldnn=False,
                ocr_version='PP-OCRv4',
                device='cpu',
            )
            with self._lock:
                self._reader = reader
                self._ready  = True
            print("[OCR-SERVICE] ✅ PaddleOCR ready!", flush=True)
        except Exception as e:
            print(f"[OCR-SERVICE] ❌ PaddleOCR init failed: {e}", flush=True)
            traceback.print_exc()
            with self._lock:
                self._failed = True

    # ── public ────────────────────────────────────────────────

    @property
    def ready(self):
        with self._lock:
            return self._ready

    @property
    def failed(self):
        with self._lock:
            return self._failed

    @property
    def stats(self):
        with self._lock:
            avg = (self._total_ms / self._calls) if self._calls else 0
            return {'calls': self._calls, 'avg_ms': round(avg, 1)}

    def run_ocr(self, img_bgr: np.ndarray) -> tuple[str, float]:
        """Run PaddleOCR on a BGR numpy array.
        Returns (raw_text, max_score) or ('', 0.0) on failure.
        """
        with self._lock:
            if not self._ready or self._reader is None:
                return ('', 0.0)
            reader = self._reader

        t0 = time.time()
        try:
            result = reader.ocr(img_bgr)
        except Exception as e:
            print(f"[OCR-SERVICE] ❌ paddle.ocr() exception: {e}", flush=True)
            traceback.print_exc()
            return ('', 0.0)
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            with self._lock:
                self._calls    += 1
                self._total_ms += elapsed_ms
            print(f"[OCR-SERVICE] OCR done in {elapsed_ms:.0f} ms", flush=True)

        # --- parse result ------------------------------------------------
        if not result or not isinstance(result, list) or not result[0]:
            return ('', 0.0)

        texts, scores = [], []
        for line in result[0]:
            if len(line) == 2 and isinstance(line[1], tuple):
                texts.append(str(line[1][0]))
                scores.append(float(line[1][1]))

        accepted = [t for t, s in zip(texts, scores) if s >= 0.25]
        raw_text = ''.join(accepted)
        max_score = max(scores) if scores else 0.0
        return (raw_text, max_score)


_engine = _OCREngine()


# ─────────────────────────────────────────────────────────────
# FLASK ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    if _engine.failed:
        return jsonify({'status': 'failed'}), 503
    if not _engine.ready:
        return jsonify({'status': 'loading'}), 202
    return jsonify({'status': 'ready'}), 200


@app.route('/status', methods=['GET'])
def status():
    state = 'failed' if _engine.failed else ('ready' if _engine.ready else 'loading')
    s = _engine.stats
    return jsonify({'status': state, **s}), 200


@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """Accept image as:
        - multipart/form-data  field "image"  (file bytes)
        - application/json     field "image_b64"  (base64 string)
        - raw body bytes       (image file bytes)
    """
    if not _engine.ready:
        return jsonify({'error': 'OCR engine not ready yet'}), 503

    # ── decode incoming image ──────────────────────────────────
    img_bgr = None
    try:
        ct = request.content_type or ''
        if 'multipart' in ct and 'image' in request.files:
            file_bytes = request.files['image'].read()
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        elif 'json' in ct:
            data = request.get_json(force=True, silent=True) or {}
            b64  = data.get('image_b64', '')
            if b64:
                file_bytes = base64.b64decode(b64)
                arr = np.frombuffer(file_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        else:
            # raw body
            file_bytes = request.get_data()
            if file_bytes:
                arr = np.frombuffer(file_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    except Exception as e:
        return jsonify({'error': f'Image decode failed: {e}'}), 400

    if img_bgr is None or img_bgr.size == 0:
        return jsonify({'error': 'No valid image received'}), 400

    # ── smart resize for best OCR accuracy ────────────────────
    h, w = img_bgr.shape[:2]
    if h < 32:
        img_bgr = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]
    elif h > 80:
        scale = 64.0 / h
        img_bgr = cv2.resize(img_bgr,
                              (max(16, int(w * scale)), 64),
                              interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    print(f"[OCR-SERVICE] /ocr  image={w}×{h}", flush=True)

    # ── run OCR ───────────────────────────────────────────────
    raw_text, score = _engine.run_ocr(img_bgr)
    print(f"[OCR-SERVICE] raw_text={repr(raw_text)}  score={score:.3f}", flush=True)

    # ── validate serial ───────────────────────────────────────
    serial = _correct_serial(raw_text) if raw_text else None
    if serial:
        print(f"[OCR-SERVICE] ✅ SERIAL={serial}", flush=True)
    else:
        print(f"[OCR-SERVICE] ⚠️  No valid serial from {repr(raw_text)}", flush=True)

    return jsonify({
        'serial':   serial,
        'raw_text': raw_text,
        'score':    round(score, 4),
    }), 200


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"[OCR-SERVICE] Starting Flask on port {OCR_SERVICE_PORT} …", flush=True)
    app.run(
        host='127.0.0.1',
        port=OCR_SERVICE_PORT,
        threaded=True,       # allow concurrent /ocr requests
        debug=False,
        use_reloader=False,
    )
