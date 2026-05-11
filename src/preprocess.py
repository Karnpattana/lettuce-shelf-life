import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """โหลดภาพ BGR → RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_with_padding(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Resize รักษา aspect ratio แล้ว pad ด้วยสีดำให้ได้ target_size x target_size"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def denoise(img: np.ndarray) -> np.ndarray:
    """Gaussian blur kernel=(5,5), sigma=1"""
    return cv2.GaussianBlur(img, (5, 5), 1)


def preprocess_pipeline(path: str, target_size: int = 512) -> np.ndarray:
    """load → resize → denoise"""
    img = load_image(path)
    img = resize_with_padding(img, target_size)
    img = denoise(img)
    return img
