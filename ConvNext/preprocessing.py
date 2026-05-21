"""
preprocessing.py — Görüntü ön işleme yardımcıları.

robust_crop : Otsu + morfoloji ile beyin bölgesini crop eder.
clahe_enhance : Kaynak domain için CLAHE (harici test'e uygulanmaz).
pad_resize    : En-boy oranını koruyarak kare pad + yeniden boyutlandırma.
process_save  : Ham görüntüyü işleyip diske kaydeder.
"""
import os
import cv2
import numpy as np
import imutils


def robust_crop(image, margin: float = 0.08):
    """Otsu eşikleme + morfoloji ile beyin bölgesini tespit edip kırpar."""
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts = imutils.grab_contours(
        cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    if not cnts:
        return image
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20:
        return image
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(image.shape[1], x + w + mx), min(image.shape[0], y + h + my)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else image


def clahe_enhance(image):
    """
    Kaynak domain görüntüleri için CLAHE kontrastı artırır.
    Harici test setine UYGULANMAZ (domain shift korunur).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def pad_resize(image, size: int = 224):
    """En-boy oranını koruyarak kare padding + yeniden boyutlandırma."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    res = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh,
           (size - nw) // 2:(size - nw) // 2 + nw] = res
    return canvas


def process_save(src: str, dst: str, apply_clahe: bool = True) -> bool:
    """
    Ham görüntüyü okur, işler (kırpma + isteğe bağlı CLAHE + yeniden boyut)
    ve belirtilen hedefe kaydeder.

    Parameters
    ----------
    src : kaynak dosya yolu
    dst : hedef dosya yolu
    apply_clahe : Kaynak domain için True, harici test için False

    Returns
    -------
    True yazma başarılı, False dosya okunamadı/hata.
    """
    img = cv2.imread(src)
    if img is None:
        return False
    img = robust_crop(img)
    if apply_clahe:
        img = clahe_enhance(img)
    img = pad_resize(img)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return cv2.imwrite(dst, img)


def preprocess_image(path: str, img_size: int = 224):
    """
    Tek bir görüntüyü yükler ve modele hazır hale getirir.

    Returns
    -------
    img_rgb  : uint8 RGB (H, W, 3)  — görselleştirme için
    img_norm : float32 [0, 1]        — model girişi için
    """
    img = cv2.imread(str(path))
    img = robust_crop(img)
    img = pad_resize(img, img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, img_rgb.astype(np.float32) / 255.0
