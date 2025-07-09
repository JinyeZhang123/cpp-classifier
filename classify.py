"""C++ Topic Classification Core Module (updated)
-------------------------------------------------
* Sliding window logic replaced with new implementation provided by user.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import os
import re
import joblib
import numpy as np

# ────────────────────────────────────────────────
# Model & assets
# ────────────────────────────────────────────────

MODEL_DIR = Path(__file__).with_suffix("").parent / "model"
MODEL_PATH = MODEL_DIR / "multi_label_svm_model.pkl"
VECT_PATH = MODEL_DIR / "vectorizer.pkl"
MLB_PATH = MODEL_DIR / "mlb.pkl"

print("[classify] Loading assets …")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
mlb = joblib.load(MLB_PATH)
print("[classify]   classes: ", ", ".join(mlb.classes_))

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

topic_window_sizes: Dict[str, int] = {
    "operator overload": 20,
    "virtual function": 40,
    "friend": 40,
    "Inheritance": 60,
    "inline": 10,
    "Namespaces": 40,
    "Recursion": 20,
    "templates": 40,
    "Classes": 20,
    "Smart_Pointers": 50,  # skipped (not in classes)
    "fpoint": 40,          # skipped (not in classes)
    "Try_Catch": 60,
}

SIZE_RATIO = 1.2

topic_colors = {t: "lightblue" for t in topic_window_sizes}

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _remove_comments(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    return code

# ╭──────── 滑动窗口 + 括号补全（新逻辑） ───────╮

def _predict_window(snippet: str, clf) -> bool:
    return clf.predict(vectorizer.transform([snippet]))[0] == 1

def sliding_window_mask(code: str, win: int, clf) -> np.ndarray:
    L = len(code)
    buf = np.zeros(L)
    for s in range(L):
        e = min(s + win, L)
        if _predict_window(code[s:e], clf):
            buf[s:e] += 1
    return (buf >= (win / SIZE_RATIO)).astype(int)

def bracket_matching_classification(code: str, topic: str, init_mask: np.ndarray) -> np.ndarray:
    L = len(code)
    classifications = [topic if init_mask[i] else None for i in range(L)]

    if topic in {"virtual function", "Namespaces"}:
        i, line_start = 0, 0
        while i < L:
            if code[i] == "\n":
                if any(classifications[j] == topic for j in range(line_start, i)):
                    for j in range(line_start, i):
                        classifications[j] = topic
                line_start = i + 1
            i += 1
        if any(classifications[j] == topic for j in range(line_start, L)):
            for j in range(line_start, L):
                classifications[j] = topic
    else:
        level, highlight = 0, False
        for i, ch in enumerate(code):
            if classifications[i] == topic:
                highlight = True
            if ch == "{" and highlight:
                level += 1
            if highlight:
                classifications[i] = topic
            if ch == "}" and highlight:
                level -= 1
                if level <= 0:
                    highlight = False

    return np.array([1 if classifications[i] == topic else 0 for i in range(L)], dtype=int)
# ╰────────────────────────────────────────────╯

# ────────────────────────────────────────────────
# HTML renderer (unchanged)
# ────────────────────────────────────────────────

def _render_html(code: str, cls: Dict[str, np.ndarray]) -> str:
    sb = []
    sb.append("<html><head><meta charset='utf-8'>")
    sb.append("""
        <script>
          function updateDisplay() {
              const pick = document.getElementById('topicSelector').value;
              document.querySelectorAll('[data-topic]').forEach(el => {
                  const topics = el.dataset.topic.split('|');
                  if (!pick || topics.includes(pick)) {
                      el.style.background = el.dataset.color || 'inherit';
                  } else {
                      el.style.background = 'transparent';
                  }
              });
          }
        </script>
        <style>pre{white-space:pre-wrap;font-family:monospace}</style>
    """)
    sb.append("</head><body>")

    # 下拉框
    sb.append('<select id="topicSelector" onchange="updateDisplay()" style="margin-bottom:10px">')
    sb.append('<option value="">All topics</option>')
    for t in topic_window_sizes:
        sb.append(f'<option value="{t}">{t}</option>')
    sb.append('</select>')

    sb.append('<pre>')
    for i, ch in enumerate(code):
        tags = [t for t, m in cls.items() if m[i]]
        color = topic_colors[tags[0]] if tags else "transparent"
        tag_str = "|".join(tags)
        # —— 所有字符都放 span，未高亮时 color=transparent —— #
        sb.append(
            f'<span data-topic="{tag_str}" data-color="{color}" '
            f'style="background:{color}">{ch}</span>'
        )
    sb.append('</pre></body></html>')
    return "".join(sb)

# ────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────

def classify_code(code: str) -> str:
    code = _remove_comments(code)
    masks: Dict[str, np.ndarray] = {}

    for topic, win in topic_window_sizes.items():
        if topic not in mlb.classes_:
            continue  # skip topics missing in model
        idx = mlb.classes_.tolist().index(topic)
        clf = model.estimators_[idx]
        init_mask = sliding_window_mask(code, win, clf)
        final_mask = bracket_matching_classification(code, topic, init_mask)
        masks[topic] = final_mask

    return _render_html(code, masks)

# ────────────────────────────────────────────────
# CLI Helper
# ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify C++ code & export HTML")
    parser.add_argument("cpp", help="Path to C++ source file")
    parser.add_argument("-o", "--output", default="output.html")
    args = parser.parse_args()

    source = Path(args.cpp).read_text(encoding="utf-8", errors="ignore")
    html = classify_code(source)
    Path(args.output).write_text(html, encoding="utf-8")
    print("[classify] HTML written →", args.output)
