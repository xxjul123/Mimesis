"""
pdi.py — Módulo de Procesamiento Digital de Imágenes (OpenCV) para MIMESIS.

Etapas implementadas:
1. Decodificación segura de imagen JPEG/PNG a matriz BGR (NumPy).
2. Control de nitidez mediante la varianza del Laplaciano (detecta fotos movidas).
3. Mejora adaptativa de iluminación y contraste con CLAHE en espacio de color LAB.
4. Redimensionamiento sin deformación (letterboxing) para no alterar ángulos corporales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class LetterboxMeta:
    orig_w: int
    orig_h: int
    target_size: int
    scale: float
    pad_x: int
    pad_y: int


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Convierte los bytes recibidos por HTTP en una imagen BGR de OpenCV."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")
    return img_bgr


def compute_sharpness(img_bgr: np.ndarray, blur_threshold: float = 40.0) -> Tuple[float, bool]:
    """
    Mide la nitidez usando la varianza del operador Laplaciano sobre escala de grises.
    Valores muy bajos indican desenfoque o movimiento fuerte.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var, lap_var < blur_threshold


def apply_adaptive_clahe(
    img_bgr: np.ndarray,
    dark_threshold: float = 115.0,
    low_contrast_std: float = 42.0,
) -> Tuple[np.ndarray, float, bool]:
    """
    Convierte a espacio LAB y analiza el canal de luminancia (L).
    Si la foto está oscura o con bajo contraste, aplica CLAHE solo sobre L
    para mejorar la visibilidad de extremidades sin distorsionar colores.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    mean_l = float(np.mean(l_channel))
    std_l = float(np.std(l_channel))
    needs_clahe = (mean_l < dark_threshold) or (std_l < low_contrast_std)

    if not needs_clahe:
        return img_bgr, mean_l, False

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr, mean_l, True


def letterbox_resize(img_bgr: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    Escala la imagen dentro de un lienzo cuadrado (target_size x target_size)
    manteniendo la relación de aspecto con relleno (padding) para no deformar ángulos.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    meta = LetterboxMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        target_size=target_size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return canvas, meta


def unletterbox_point(x_norm: float, y_norm: float, meta: LetterboxMeta) -> Tuple[float, float]:
    """
    Transforma una coordenada normalizada [0, 1] del lienzo letterbox de vuelta
    a la coordenada [0, 1] de la fotografía original.
    """
    px = x_norm * meta.target_size
    py = y_norm * meta.target_size
    orig_x = (px - meta.pad_x) / (meta.scale * meta.orig_w)
    orig_y = (py - meta.pad_y) / (meta.scale * meta.orig_h)
    return float(np.clip(orig_x, 0.0, 1.0)), float(np.clip(orig_y, 0.0, 1.0))


def preprocess_for_pose(image_bytes: bytes, target_size: int = 640) -> Tuple[np.ndarray, LetterboxMeta, Dict]:
    """
    Ejecuta el pipeline completo de preprocesamiento PDI:
    decodificación -> nitidez (Laplaciano) -> CLAHE adaptativo (LAB) -> letterboxing -> RGB.
    """
    img_bgr = decode_image(image_bytes)
    lap_var, is_blurry = compute_sharpness(img_bgr)
    enhanced_bgr, mean_l, clahe_applied = apply_adaptive_clahe(img_bgr)
    boxed_bgr, meta = letterbox_resize(enhanced_bgr, target_size=target_size)
    boxed_rgb = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)

    metrics = {
        "laplacian_var": round(lap_var, 2),
        "is_blurry": is_blurry,
        "mean_luminance": round(mean_l, 2),
        "clahe_applied": clahe_applied,
        "orig_size": [meta.orig_w, meta.orig_h],
        "target_size": target_size,
    }
    return boxed_rgb, meta, metrics
