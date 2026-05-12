import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_texture(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    สกัด GLCM texture features จาก L* channel ของ pixel ในใบ

    Parameters
    ----------
    img_rgb : H×W×3 uint8 (RGB)
    mask    : H×W uint8 (0/255)

    Returns
    -------
    dict: contrast, correlation, energy, homogeneity
    """
    import cv2
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]  # L channel [0,255]

    # ใส่ pixel นอกใบเป็น 0 แล้ว crop bounding box เพื่อลดขนาด matrix
    masked_L = np.where(mask > 0, L, 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return _empty_texture_features()

    roi = masked_L[ys.min():ys.max()+1, xs.min():xs.max()+1]

    glcm = graycomatrix(
        roi,
        distances=[5],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    return {
        "contrast":    float(graycoprops(glcm, "contrast").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
        "energy":      float(graycoprops(glcm, "energy").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
    }


def _empty_texture_features() -> dict:
    return {k: float("nan") for k in
            ["contrast", "correlation", "energy", "homogeneity"]}
