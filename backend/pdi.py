"""
pdi.py — Módulo de Procesamiento Digital de Imágenes (OpenCV) para MIMESIS.

Etapas implementadas:
1. Decodificación segura de imagen JPEG/PNG a matriz BGR (NumPy).
2. Control de nitidez mediante la varianza del Laplaciano (detecta fotos movidas).
3. Mejora adaptativa de iluminación y contraste con CLAHE en espacio de color LAB.
4. Redimensionamiento sin deformación (letterboxing) para no alterar ángulos corporales.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def unletterbox_mask(mask: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """
    Revierte el letterboxing sobre una máscara 2D (float o uint8) devolviéndola
    a las dimensiones exactas (orig_h, orig_w) de la fotografía original.
    """
    mh, mw = mask.shape[:2]
    scale_x = mw / float(meta.target_size)
    scale_y = mh / float(meta.target_size)

    new_w = int(round(meta.orig_w * meta.scale))
    new_h = int(round(meta.orig_h * meta.scale))

    x0 = max(0, int(round(meta.pad_x * scale_x)))
    y0 = max(0, int(round(meta.pad_y * scale_y)))
    x1 = min(mw, int(round((meta.pad_x + new_w) * scale_x)))
    y1 = min(mh, int(round((meta.pad_y + new_h) * scale_y)))

    cropped = mask[y0:y1, x0:x1]
    if cropped.size == 0:
        cropped = mask

    interp = cv2.INTER_NEAREST if mask.dtype == np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(cropped, (meta.orig_w, meta.orig_h), interpolation=interp)


def preprocess_with_intermediates(
    image_bytes: bytes,
    target_size: int = 640,
) -> Tuple[np.ndarray, LetterboxMeta, Dict, np.ndarray, np.ndarray]:
    """
    Ejecuta el pipeline completo de preprocesamiento PDI devolviendo además
    las imágenes intermedias (img_bgr original y enhanced_bgr tras CLAHE)
    para el Inspector Visual PDI y la vista Modo Escultura.
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
    return boxed_rgb, meta, metrics, img_bgr, enhanced_bgr


def preprocess_for_pose(image_bytes: bytes, target_size: int = 640) -> Tuple[np.ndarray, LetterboxMeta, Dict]:
    """
    Ejecuta el pipeline completo de preprocesamiento PDI:
    decodificación -> nitidez (Laplaciano) -> CLAHE adaptativo (LAB) -> letterboxing -> RGB.
    """
    boxed_rgb, meta, metrics, _, _ = preprocess_with_intermediates(image_bytes, target_size=target_size)
    return boxed_rgb, meta, metrics


def _resize_max_side(img: np.ndarray, max_side: int = 440, interp: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _encode_jpeg_data_url(img_bgr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def generate_pdi_debug_views(
    img_bgr: np.ndarray,
    enhanced_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_side: int = 440,
) -> Dict[str, Optional[str]]:
    """
    Genera vistas livianas (JPEG base64) de las etapas reales del pipeline PDI:
    1. clahe_view: imagen tras mejora adaptativa CLAHE en canal L (LAB).
    2. laplacian_view: mapa normalizado de bordes del operador Laplaciano.
    3. mask_contour_view: máscara binaria morfológica + contorno principal (findContours).
    4. sculpture_mode_view: silueta segmentada con estética monocroma/bronce sobre
       fondo de galería oscurecido y desenfocado.
    """
    orig_small = _resize_max_side(img_bgr, max_side=max_side)
    enh_small = _resize_max_side(enhanced_bgr, max_side=max_side)

    # 1. Vista CLAHE
    clahe_view = _encode_jpeg_data_url(enh_small, quality=82)

    # 2. Mapa Laplaciano normalizado (evidencia visual de nitidez / bordes)
    gray = cv2.cvtColor(orig_small, cv2.COLOR_BGR2GRAY)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    p99 = float(np.percentile(lap_abs, 99.0))
    scale_div = p99 if p99 > 1e-3 else 1.0
    lap_norm = np.clip((lap_abs / scale_div) * 255.0, 0, 255).astype(np.uint8)
    lap_colored = cv2.applyColorMap(lap_norm, cv2.COLORMAP_INFERNO)
    laplacian_view = _encode_jpeg_data_url(lap_colored, quality=82)

    mask_contour_view: Optional[str] = None
    sculpture_mode_view: Optional[str] = None

    if mask is not None and mask.size > 0:
        h, w = orig_small.shape[:2]
        mask_small = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Vista de máscara binaria + contorno principal (cv2.findContours)
        canvas = np.full((h, w, 3), (18, 17, 15), dtype=np.uint8)
        canvas[mask_small > 127] = (232, 224, 212)  # Marfil piedra
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(canvas, [main_cnt], -1, (85, 141, 200), 2, cv2.LINE_AA)  # Bronce BGR
        mask_contour_view = _encode_jpeg_data_url(canvas, quality=84)

        # 4. Vista "Modo Escultura" (silueta segmentada en tono galería + fondo desenfocado)
        blur_k = max(15, (min(h, w) // 12) | 1)
        bg_blur = cv2.GaussianBlur(orig_small, (blur_k, blur_k), 0)
        bg_dark = (bg_blur.astype(np.float32) * 0.24).astype(np.uint8)

        # Estética piedra/bronce cálido en la persona segmentada
        fg_gray = cv2.cvtColor(enh_small, cv2.COLOR_BGR2GRAY)
        fg_clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(fg_gray)
        fg_bronze = np.stack(
            [
                np.clip(fg_clahe * 0.78, 0, 255),  # B
                np.clip(fg_clahe * 0.90, 0, 255),  # G
                np.clip(fg_clahe * 1.04, 0, 255),  # R
            ],
            axis=-1,
        ).astype(np.uint8)
        fg_styled = cv2.addWeighted(enh_small, 0.32, fg_bronze, 0.68, 0)

        # Suavizado de bordes de la máscara para transición limpia
        alpha = cv2.GaussianBlur(mask_small.astype(np.float32) / 255.0, (7, 7), 0)[..., None]
        composed = (fg_styled.astype(np.float32) * alpha + bg_dark.astype(np.float32) * (1.0 - alpha))
        composed_u8 = np.clip(composed, 0, 255).astype(np.uint8)

        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(composed_u8, [main_cnt], -1, (110, 165, 218), 1, cv2.LINE_AA)

        sculpture_mode_view = _encode_jpeg_data_url(composed_u8, quality=86)

    return {
        "clahe_view": clahe_view,
        "laplacian_view": laplacian_view,
        "mask_contour_view": mask_contour_view,
        "sculpture_mode_view": sculpture_mode_view,
    }
