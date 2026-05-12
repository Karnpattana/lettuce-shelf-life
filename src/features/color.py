import cv2
import numpy as np

from src.features import (
    BROWN_LAB_RANGE,
    GREEN_LAB_RANGE,
    YELLOW_LAB_RANGE,
)


def extract_color(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    สกัด color features จาก pixel ในใบเท่านั้น (mask > 0)

    Parameters
    ----------
    img_rgb : H×W×3 uint8 (RGB)
    mask    : H×W uint8 (0/255)

    Returns
    -------
    dict ของ color features
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV LAB: L [0,255] → /255*100, a/b [0,255] → -128..127
    lab[:, :, 0] = lab[:, :, 0] / 255.0 * 100.0
    lab[:, :, 1] = lab[:, :, 1] - 128.0
    lab[:, :, 2] = lab[:, :, 2] - 128.0

    px = mask > 0
    if px.sum() == 0:
        return _empty_color_features()

    L = lab[:, :, 0][px]
    a = lab[:, :, 1][px]
    b = lab[:, :, 2][px]
    n = len(L)

    pct_green = np.mean(
        (a < GREEN_LAB_RANGE["a_max"]) & (L > GREEN_LAB_RANGE["L_min"])
    )
    pct_yellow = np.mean(
        (a >= YELLOW_LAB_RANGE["a_min"])
        & (a <= YELLOW_LAB_RANGE["a_max"])
        & (b > YELLOW_LAB_RANGE["b_min"])
    )
    pct_brown = np.mean(
        (a > BROWN_LAB_RANGE["a_min"])
        & (b > BROWN_LAB_RANGE["b_min"])
        & (L < BROWN_LAB_RANGE["L_max"])
    )

    return {
        "L_mean": float(np.mean(L)),
        "L_std":  float(np.std(L)),
        "a_mean": float(np.mean(a)),
        "a_std":  float(np.std(a)),
        "b_mean": float(np.mean(b)),
        "b_std":  float(np.std(b)),
        "pct_green":  float(pct_green),
        "pct_yellow": float(pct_yellow),
        "pct_brown":  float(pct_brown),
    }


def _empty_color_features() -> dict:
    return {k: float("nan") for k in [
        "L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std",
        "pct_green", "pct_yellow", "pct_brown",
    ]}
