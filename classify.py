"""C++ Topic Classification Core Module (param‑tunable)
-------------------------------------------------------
* Window sizes can be globally scaled with ``win_scale``.
* Threshold ``SIZE_RATIO`` can be passed per request.
* Public API: ``classify_code(code, size_ratio=1.2, win_scale=1.0)``.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

# ───────────────── assets ─────────────────
MODEL_DIR = Path(__file__).with_suffix("").parent / "model"
model = joblib.load(MODEL_DIR / "multi_label_svm_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
mlb        = joblib.load(MODEL_DIR / "mlb.pkl")
print("[classify] classes:", ", ".join(mlb.classes_))

# ─────────────── configuration ───────────
_base_window_sizes: Dict[str, int] = {
    "operator overload": 20,
    "virtual function": 40,
    "friend": 40,
    "Inheritance": 60,
    "inline": 10,
    "templates": 40,
    "Classes": 20,
    "Try_Catch": 60,
}

default_colors = {t: "lightblue" for t in _base_window_sizes}

# ─────────────── helper funcs ─────────────

def _remove_comments(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    return code

def _predict_window(snippet: str, clf) -> bool:
    return clf.predict(vectorizer.transform([snippet]))[0] == 1

def sliding_window_mask(code: str, win: int, clf, ratio: float) -> np.ndarray:
    L = len(code)
    buf = np.zeros(L)
    for s in range(L):
        e = min(s + win, L)
        if _predict_window(code[s:e], clf):
            buf[s:e] += 1
    return (buf >= (win / ratio)).astype(int)

def bracket_spread(code: str, topic: str, init: np.ndarray) -> np.ndarray:
    L = len(code)
    labels = [topic if init[i] else None for i in range(L)]
    lvl, on = 0, False
    for i, ch in enumerate(code):
        if labels[i] == topic:
            on = True
        if ch == "{" and on:
            lvl += 1
        if on:
            labels[i] = topic
        if ch == "}" and on:
            lvl -= 1
            if lvl <= 0:
                on = False
    return np.array([1 if labels[i] == topic else 0 for i in range(L)], dtype=int)

# ─────────────── HTML render ──────────────

def _render_html(code: str, masks: Dict[str, np.ndarray], colors=default_colors) -> str:
    sb: List[str] = []
    sb.append("<html><head><meta charset='utf-8'>")
    sb.append("""
    <script>
    function updateDisplay(){
      const pick=document.getElementById('topicSelector').value;
      document.querySelectorAll('[data-topic]').forEach(el=>{
        const topics=el.dataset.topic.split('|');
        el.style.background = (!pick||topics.includes(pick)) ? el.dataset.color : 'transparent';
      });
    }
    </script>
    <style>pre{white-space:pre-wrap;font-family:monospace}</style>
    """)
    sb.append("</head><body>")
    sb.append('<select id="topicSelector" onchange="updateDisplay()" style="margin-bottom:10px">')
    sb.append('<option value="">All topics</option>')
    for t in _base_window_sizes:
        sb.append(f'<option value="{t}">{t}</option>')
    sb.append('</select><pre>')
    for i, ch in enumerate(code):
        tags=[t for t,m in masks.items() if m[i]]
        color=colors.get(tags[0],"transparent") if tags else "transparent"
        sb.append(f'<span data-topic="{"|".join(tags)}" data-color="{color}" style="background:{color}">{ch}</span>')
    sb.append('</pre></body></html>')
    return "".join(sb)

# ─────────────── public API ───────────────

def classify_code(
    code: str,
    size_ratio: float = 1.2,
    custom_windows: Dict[str, int] | None = None,
) -> str:
    """Main entry.

    Parameters
    ----------
    code : str
        C++ source string.
    size_ratio : float, default 1.2
        Threshold ratio for sliding-window vote.
    custom_windows : Optional[Dict[str, int]]
        Per‑topic window sizes; missing keys fall back to defaults.
    """
    code = _remove_comments(code)

    # merge window sizes
    win_sizes = _base_window_sizes.copy()
    if custom_windows:
        for k, v in custom_windows.items():
            if k in win_sizes and v > 0:
                win_sizes[k] = int(v)

    masks: Dict[str, np.ndarray] = {}
    for topic, win in win_sizes.items():
        if topic not in mlb.classes_:
            continue
        clf = model.estimators_[mlb.classes_.tolist().index(topic)]
        init = sliding_window_mask(code, win, clf, size_ratio)
        masks[topic] = bracket_spread(code, topic, init)

    return _render_html(code, masks)
