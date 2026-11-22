"""
pose.py — Estimación de 33 landmarks (MediaPipe BlazePose) y
normalización geométrica invariante a traslación y escala.

Soporta automáticamente:
- MediaPipe moderno (0.10.30+ / 1.x): Tasks API (PoseLandmarker)
- MediaPipe clásico (<= 0.10.21): mp.solutions.pose
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pdi import LetterboxMeta, unletterbox_point

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
TASK_MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_full.task"


# 12 ángulos clave agrupados por región corporal (en grados, 0..180)
REGION_ANGLES: Dict[str, List[str]] = {
    "head": ["head_tilt", "neck_angle"],
    "torso": ["torso_lean", "shoulder_level"],
    "arms": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
    "legs": ["left_hip", "right_hip", "left_knee", "right_knee"],
}

FEATURE_ORDER: List[str] = [
    "head_tilt",
    "neck_angle",
    "torso_lean",
    "shoulder_level",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calcula el ángulo ABC en grados (0..180)."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_val)))


def _line_angle_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    """Ángulo en grados respecto al eje vertical (0 = completamente erguido)."""
    vec = p_top - p_bottom
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0
    vertical = np.array([0.0, -1.0])
    cos_val = np.clip(np.dot(vec, vertical) / norm, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_val)))


def _line_angle_horizontal(p_left: np.ndarray, p_right: np.ndarray) -> float:
    """Inclinación respecto a la horizontal en grados (0 = perfectamente nivelado)."""
    vec = p_right - p_left
    return float(abs(math.degrees(math.atan2(vec[1], abs(vec[0]) + 1e-6))))


def _compute_head_tilt(landmarks: List[Dict[str, float]], pts: np.ndarray) -> float:
    """
    Calcula la inclinación de la cabeza (0..90°):
    - Vista frontal (ambas orejas 7 y 8 visibles y separadas horizontalmente):
      usa la línea entre Oreja I (7) y Oreja D (8).
    - Vista de perfil / 3/4 (una oreja oculta o comprimida horizontalmente):
      usa la línea desde la oreja visible hacia la nariz (0) para evitar ruido.
    """
    vis_l = float(landmarks[7].get("visibility", 1.0)) if len(landmarks) > 7 else 0.0
    vis_r = float(landmarks[8].get("visibility", 1.0)) if len(landmarks) > 8 else 0.0

    shoulder_width = float(np.linalg.norm(pts[11] - pts[12]))
    ear_dx = float(abs(pts[7][0] - pts[8][0]))

    both_ears_clear = (
        vis_l >= 0.45
        and vis_r >= 0.45
        and (shoulder_width < 1e-4 or ear_dx >= 0.20 * shoulder_width)
    )
    if both_ears_clear:
        return _line_angle_horizontal(pts[7], pts[8])

    best_ear = 7 if vis_l >= vis_r else 8
    return _line_angle_horizontal(pts[best_ear], pts[0])


def extract_geometric_features(landmarks: List[Dict[str, float]]) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Convierte los 33 landmarks normalizados en:
    1) Diccionario de 12 ángulos corporales invariantes a escala, posición y fondo.
    2) Vector NumPy normalizado [0, 1] listo para el MLP.
    """
    pts = np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float32)

    mid_shoulder = (pts[11] + pts[12]) / 2.0
    mid_hip = (pts[23] + pts[24]) / 2.0

    angles: Dict[str, float] = {
        "head_tilt": _compute_head_tilt(landmarks, pts),
        "neck_angle": _line_angle_vertical(pts[0], mid_shoulder),
        "torso_lean": _line_angle_vertical(mid_shoulder, mid_hip),
        "shoulder_level": _line_angle_horizontal(pts[11], pts[12]),
        "left_elbow": _angle_3pts(pts[11], pts[13], pts[15]),
        "right_elbow": _angle_3pts(pts[12], pts[14], pts[16]),
        "left_shoulder": _angle_3pts(pts[23], pts[11], pts[13]),
        "right_shoulder": _angle_3pts(pts[24], pts[12], pts[14]),
        "left_hip": _angle_3pts(pts[11], pts[23], pts[25]),
        "right_hip": _angle_3pts(pts[12], pts[24], pts[26]),
        "left_knee": _angle_3pts(pts[23], pts[25], pts[27]),
        "right_knee": _angle_3pts(pts[24], pts[26], pts[28]),
    }

    feature_vec = np.array([angles[name] / 180.0 for name in FEATURE_ORDER], dtype=np.float32)
    return angles, feature_vec


def _ensure_task_model() -> Path:
    """Descarga el modelo BlazePose (.task) una sola vez si no existe en backend/models/."""
    TASK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TASK_MODEL_PATH.exists():
        print(f"[MIMESIS] Descargando modelo BlazePose a {TASK_MODEL_PATH}...")
        urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)
    return TASK_MODEL_PATH


class PoseExtractor:
    """Envoltorio reutilizable de MediaPipe BlazePose para imágenes estáticas."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.45):
        if mp is None:
            raise RuntimeError(
                "MediaPipe no está instalado. Instala con: pip install -r backend/requirements.txt"
            )

        # Compatibilidad con MediaPipe clásico (mp.solutions) y moderno (mp.tasks)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._backend = "solutions"
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            self._backend = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_task_model()
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_buffer=model_path.read_bytes()
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def extract(self, boxed_rgb: np.ndarray, meta: LetterboxMeta) -> Optional[Tuple[List[Dict[str, float]], float]]:
        """
        Devuelve (landmarks_en_coordenadas_originales, visibilidad_promedio)
        o None si no se detectó ninguna persona.
        """
        if self._backend == "solutions":
            result = self._pose.process(boxed_rgb)
            if not result.pose_landmarks:
                return None
            raw_lms = result.pose_landmarks.landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(boxed_rgb))
            result = self._landmarker.detect(mp_image)
            if not result.pose_landmarks:
                return None
            raw_lms = result.pose_landmarks[0]

        landmarks: List[Dict[str, float]] = []
        visibilities: List[float] = []
        for lm in raw_lms:
            orig_x, orig_y = unletterbox_point(lm.x, lm.y, meta)
            vis = float(getattr(lm, "visibility", 1.0))
            visibilities.append(vis)
            landmarks.append(
                {
                    "x": round(orig_x, 5),
                    "y": round(orig_y, 5),
                    "z": round(float(lm.z), 5),
                    "visibility": round(vis, 4),
                }
            )

        mean_vis = float(np.mean(visibilities)) if visibilities else 0.0
        return landmarks, mean_vis
