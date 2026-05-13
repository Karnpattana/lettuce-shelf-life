"""ทดสอบ segment กับภาพ 12 ตัวอย่าง (D0,D3,D6,D8) แล้ว save comparison grid"""
import sys
sys.path.insert(0, '.')

import random
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocess import preprocess_pipeline
from src.segment import segment_lettuce

RAW_DIR = Path("data/raw")
PATTERN = re.compile(r'^(COS|GOK)(\d{2})_([AB])_D(\d+)_([ME])_(top|side)\.jpg$', re.IGNORECASE)

random.seed(99)
by_day = {}
for f in RAW_DIR.glob("*.jpg"):
    m = PATTERN.match(f.name)
    if m:
        by_day.setdefault(int(m.group(4)), []).append(f)

# เลือก 3 ภาพ จาก D0, D3, D6, D8
sample = []
for d in [0, 3, 6, 8]:
    if d in by_day:
        sample += random.sample(by_day[d], min(3, len(by_day[d])))

fig, axes = plt.subplots(len(sample), 3, figsize=(12, len(sample) * 2.8))
for i, img_path in enumerate(sample):
    img = preprocess_pipeline(str(img_path))
    mask, cropped = segment_lettuce(img)
    area_ratio = (mask > 0).sum() / (512 * 512)
    for ax, data, title in zip(axes[i],
                               [img, mask, cropped],
                               ['Original', f'Mask ({area_ratio:.1%})', 'Cropped']):
        ax.imshow(data, cmap='gray' if title.startswith('Mask') else None)
        ax.set_title(f"{title}\n{img_path.name}", fontsize=6)
        ax.axis('off')

plt.suptitle('Segment QC — V_MIN=15, closing=15x15', fontsize=11)
plt.tight_layout()
out = Path("results/segment_qc_tuned.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=90, bbox_inches='tight')
print(f"Saved: {out}")
