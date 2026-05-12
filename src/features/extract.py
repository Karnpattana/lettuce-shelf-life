import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.color import extract_color
from src.features.texture import extract_texture

FILENAME_PATTERN = re.compile(
    r'^(COS|GOK)(\d{2})_([AB])_D(\d+)_([ME])_(top|side)$',
    re.IGNORECASE
)


def _parse_filename(stem: str) -> dict | None:
    m = FILENAME_PATTERN.match(stem)
    if not m:
        return None
    return {
        "variety":   m.group(1).upper(),
        "plant_id":  int(m.group(2)),
        "ab_group":  m.group(3).upper(),
        "day":       int(m.group(4)),
        "session":   m.group(5).upper(),
        "view":      m.group(6).lower(),
    }


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    # group เช่น "01-10" หรือ "11-20" → ใช้เป็น key ร่วมกับ day + session
    return df


def _match_temp(meta: pd.DataFrame, plant_id: int, day: int, session: str):
    """หา temperature row ที่ตรงกับ plant_id / day / session"""
    for _, row in meta.iterrows():
        # parse group range เช่น "01-10"
        parts = str(row["group"]).split("-")
        lo, hi = int(parts[0]), int(parts[1])
        if lo <= plant_id <= hi and int(row["day"]) == day and row["session"].upper() == session:
            try:
                return float(row["temperature_min"]), float(row["temperature_max"])
            except (ValueError, TypeError):
                return float("nan"), float("nan")
    return float("nan"), float("nan")


def extract_all(
    segmented_dir: str | Path = "data/segmented",
    metadata_path: str | Path = "data/metadata.csv",
    output_path:   str | Path = "data/features.csv",
) -> pd.DataFrame:
    segmented_dir = Path(segmented_dir)
    cropped_dir = segmented_dir / "cropped"
    masks_dir   = segmented_dir / "masks"

    meta = _load_metadata(Path(metadata_path))
    rows = []

    cropped_files = sorted(cropped_dir.glob("*_cropped.jpg"))
    for crop_path in tqdm(cropped_files, desc="Extracting features"):
        # stem เช่น "COS01_A_D0_E_side_cropped" → ตัด _cropped
        stem = crop_path.stem.replace("_cropped", "")
        mask_path = masks_dir / f"{stem}_mask.png"

        parsed = _parse_filename(stem)
        if parsed is None:
            continue

        cropped = cv2.cvtColor(cv2.imread(str(crop_path)), cv2.COLOR_BGR2RGB)
        mask    = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() \
                  else np.zeros(cropped.shape[:2], dtype=np.uint8)

        area_ratio = float(np.sum(mask > 0)) / mask.size

        color_feats   = extract_color(cropped, mask)
        texture_feats = extract_texture(cropped, mask)

        temp_min, temp_max = _match_temp(
            meta, parsed["plant_id"], parsed["day"], parsed["session"]
        )

        row = {
            "filename":  crop_path.name.replace("_cropped.jpg", ".jpg"),
            **parsed,
            **color_feats,
            **texture_feats,
            "area_ratio": area_ratio,
            "temp_min":   temp_min,
            "temp_max":   temp_max,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"NaN count:\n{df.isna().sum()[df.isna().sum() > 0]}")
    return df


if __name__ == "__main__":
    extract_all()
