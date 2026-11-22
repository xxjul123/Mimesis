"""
segmentation.py — Módulo de segmentación de silueta, morfología matemática (OpenCV)
y comparación de forma (IoU alineado + Momentos de Hu) para MIMESIS.

Etapas:
1. Umbralización + apertura/cierre morfológico (cv2.morphologyEx) para limpiar bordes
   y rellenar pequeños huecos internos.
2. Conservación del componente conexo principal (la persona/estatua) para eliminar
   falsos positivos del fondo.
3. Extracción del contorno principal (cv2.findContours).
4. Normalización de silueta preservando relación de aspecto (128x128) y comparación
   contra la máscara de referencia mediante IoU y Momentos de Hu (normal y espejo).
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def refine_mask(raw_mask: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """
    Convierte una máscara de probabilidad [0, 1] o uint8 [0, 255] en una máscara
    binaria limpia (uint8: 0 o 255) aplicando morfología matemática y conservando
    el componente conexo de mayor área.
    """
    if raw_mask.dtype != np.uint8:
        binary = (raw_mask >= threshold).astype(np.uint8) * 255
    else:
        thresh_val = int(round(threshold * 255)) if threshold <= 1.0 else int(threshold)
        _, binary = cv2.threshold(raw_mask, thresh_val, 255, cv2.THRESH_BINARY)

    h, w = binary.shape[:2]
    k = max(3, int(round(min(h, w) * 0.012)) | 1)
    k = min(k, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Apertura elimina ruido aislado; cierre rellena pequeños huecos internos
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Conservar únicamente el componente conexo más grande (persona principal)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return closed

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return clean


def extract_main_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Extrae el contorno externo de mayor área mediante cv2.findContours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def normalize_silhouette(mask: np.ndarray, size: int = 128) -> Optional[np.ndarray]:
    """
    Recorta la silueta a su bounding box ajustado y la centra en un lienzo cuadrado
    (size x size) preservando su relación de aspecto para comparar formas sin distorsión.
    """
    ys, xs = np.where(mask > 127)
    if len(xs) < 20:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1]

    ch, cw = crop.shape[:2]
    scale = min((size - 8) / max(cw, 1), (size - 8) / max(ch, 1))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))

    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    px = (size - nw) // 2
    py = (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = (resized > 127).astype(np.uint8) * 255
    return canvas


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calcula Intersection over Union (IoU) entre dos máscaras binarias del mismo tamaño."""
    a = mask_a > 127
    b = mask_b > 127
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return float(inter / union)


def compute_hu_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Calcula similitud de forma (0..100) usando Momentos de Hu (cv2.matchShapes)
    sobre los contornos principales de dos siluetas normalizadas.
    """
    cnt_a = extract_main_contour(mask_a)
    cnt_b = extract_main_contour(mask_b)
    if cnt_a is None or cnt_b is None:
        return 0.0
    if cv2.contourArea(cnt_a) < 10 or cv2.contourArea(cnt_b) < 10:
        return 0.0

    dist = float(cv2.matchShapes(cnt_a, cnt_b, cv2.CONTOURS_MATCH_I1, 0.0))
    sim = 100.0 * float(np.exp(-min(dist, 5.0) * 1.8))
    return round(float(np.clip(sim, 0.0, 100.0)), 1)


def compare_silhouettes(player_mask: Optional[np.ndarray], ref_mask: Optional[np.ndarray]) -> Optional[Dict]:
    """
    Compara la silueta segmentada del jugador contra la máscara de referencia
    de la escultura (evaluando orientación normal y espejo horizontal).
    """
    if player_mask is None or ref_mask is None:
        return None

    norm_player = normalize_silhouette(player_mask, size=128)
    norm_player_mirror = normalize_silhouette(cv2.flip(player_mask, 1), size=128)
    norm_ref = normalize_silhouette(ref_mask, size=128)
    if norm_player is None or norm_player_mirror is None or norm_ref is None:
        return None

    iou_normal = compute_iou(norm_player, norm_ref)
    iou_mirror = compute_iou(norm_player_mirror, norm_ref)

    if iou_mirror > iou_normal:
        best_iou = iou_mirror
        best_hu = compute_hu_similarity(norm_player_mirror, norm_ref)
        mirrored = True
    else:
        best_iou = iou_normal
        best_hu = compute_hu_similarity(norm_player, norm_ref)
        mirrored = False

    fg_ratio = float((player_mask > 127).mean()) * 100.0
    return {
        "iou": round(best_iou * 100.0, 1),
        "hu_similarity": best_hu,
        "foreground_ratio": round(fg_ratio, 1),
        "mirrored": mirrored,
    }
