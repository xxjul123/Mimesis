"""
evaluator.py — Evaluador de similitud de poses para MIMESIS.

Combina:
1. Evaluación por región corporal (cabeza, torso, brazos, piernas) usando diferencias
   angulares frente a la pose de referencia (reference_poses.json), soportando espejo
   izquierda/derecha para que el jugador pueda imitar hacia cualquiera de los dos lados.
2. Modelo MLP propio en PyTorch (backend/models/pose_mlp.pt) entrenado con las 480 fotos
   reales del Parcial 2. Si todavía no existe el archivo .pt, usa el puntaje geométrico
   ponderado para poder probar la aplicación de extremo a extremo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose import FEATURE_ORDER, REGION_ANGLES

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

DATA_PATH = Path(__file__).parent / "data" / "reference_poses.json"
MODEL_PATH = Path(__file__).parent / "models" / "pose_mlp.pt"

SCULPTURE_CLASSES = ["belalcazar", "discobolo", "shakespeare"]
REGION_WEIGHTS = {"head": 0.12, "torso": 0.25, "arms": 0.38, "legs": 0.25}


if nn is not None:

    class PoseMLP(nn.Module):
        """
        Perceptrón Multicapa (MLP) propio para clasificar y puntuar poses a partir
        de los 12 ángulos corporales normalizados [0, 1].
        """

        def __init__(self, in_features: int = 12, num_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            return self.net(x)


def _mirror_angles(angles: Dict[str, float]) -> Dict[str, float]:
    """Intercambia izquierdo/derecho para aceptar poses hechas en espejo."""
    mirrored = dict(angles)
    for pair in (
        ("left_elbow", "right_elbow"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
    ):
        mirrored[pair[0]], mirrored[pair[1]] = angles[pair[1]], angles[pair[0]]
    return mirrored


def _region_scores(angles: Dict[str, float], ref: Dict[str, float]) -> Dict[str, float]:
    """Calcula similitud 0..100 por cada una de las 4 regiones corporales."""
    # Tolerancias angulares (grados de diferencia media para caer a ~37% con exp)
    tolerances = {"head": 28.0, "torso": 28.0, "arms": 42.0, "legs": 42.0}
    scores: Dict[str, float] = {}
    for region, keys in REGION_ANGLES.items():
        diffs = [abs(angles[k] - ref[k]) for k in keys if k in angles and k in ref]
        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        sim = 100.0 * float(np.exp(-((mean_diff / tolerances[region]) ** 1.4)))
        scores[region] = round(float(np.clip(sim, 0.0, 100.0)), 1)
    return scores


def _weighted_total(regions: Dict[str, float]) -> float:
    return sum(regions[r] * REGION_WEIGHTS[r] for r in REGION_WEIGHTS)


def _build_feedback(regions: Dict[str, float], score: float) -> str:
    labels = {"head": "la cabeza", "torso": "el torso", "arms": "los brazos", "legs": "las piernas"}
    weakest = min(regions, key=lambda r: regions[r])
    strongest = max(regions, key=lambda r: regions[r])
    if score >= 85:
        return f"Excelente imitación: {labels[strongest]} coinciden casi exacto con la escultura."
    if score >= 60:
        return f"Buena postura general. Para subir tu puntaje, ajusta un poco más {labels[weakest]}."
    return f"Aún hay diferencia notable en {labels[weakest]}. Observa la obra de referencia e inténtalo de nuevo."


class PoseEvaluator:
    def __init__(self):
        self.references: Dict[str, Dict[str, float]] = {}
        self.reference_landmarks: Dict[str, List[Dict[str, float]]] = {}
        self._load_references()
        self.mlp = None
        self.mlp_loaded = False
        self._try_load_mlp()

    def _load_references(self) -> None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            if "angles" in v and isinstance(v["angles"], dict):
                self.references[k] = v["angles"]
                if isinstance(v.get("landmarks"), list):
                    self.reference_landmarks[k] = v["landmarks"]
            else:
                self.references[k] = v

    def _try_load_mlp(self) -> None:
        if torch is None or not MODEL_PATH.exists():
            return
        try:
            model = PoseMLP(in_features=len(FEATURE_ORDER), num_classes=len(SCULPTURE_CLASSES))
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state)
            model.eval()
            self.mlp = model
            self.mlp_loaded = True
        except Exception as exc:  # pragma: no cover
            print(f"[MIMESIS] Aviso: no se pudo cargar {MODEL_PATH}: {exc}")
            self.mlp = None
            self.mlp_loaded = False

    def _mlp_probability(self, angles: Dict[str, float], sculpture_id: str) -> Optional[float]:
        if not self.mlp_loaded or self.mlp is None or torch is None:
            return None
        if sculpture_id not in SCULPTURE_CLASSES:
            return None

        target_idx = SCULPTURE_CLASSES.index(sculpture_id)
        vec_normal = np.array([angles[k] / 180.0 for k in FEATURE_ORDER], dtype=np.float32)
        mirrored = _mirror_angles(angles)
        vec_mirror = np.array([mirrored[k] / 180.0 for k in FEATURE_ORDER], dtype=np.float32)

        with torch.no_grad():
            x = torch.from_numpy(np.stack([vec_normal, vec_mirror], axis=0))
            probs = torch.softmax(self.mlp(x), dim=1)[:, target_idx].numpy()
        return float(np.max(probs))

    def evaluate(self, angles: Dict[str, float], sculpture_id: str) -> Dict:
        sid = (sculpture_id or "belalcazar").lower()
        ref = self.references.get(sid) or next(iter(self.references.values()))

        # Evalúa orientación normal y en espejo, y conserva la mejor coincidencia
        reg_normal = _region_scores(angles, ref)
        reg_mirror = _region_scores(_mirror_angles(angles), ref)

        if _weighted_total(reg_mirror) > _weighted_total(reg_normal):
            regions = reg_mirror
        else:
            regions = reg_normal

        geo_score = _weighted_total(regions)
        mlp_prob = self._mlp_probability(angles, sid)

        if mlp_prob is not None:
            # Combina probabilidad del MLP propio (60%) con precisión geométrica por zona (40%)
            final_score = 0.60 * (mlp_prob * 100.0) + 0.40 * geo_score
            mode = "mlp"
        else:
            final_score = geo_score
            mode = "geometric"

        final_score = round(float(np.clip(final_score, 0.0, 100.0)), 1)
        feedback = _build_feedback(regions, final_score)

        return {
            "score": final_score,
            "regions": regions,
            "feedback": feedback,
            "evaluator_mode": mode,
            "mlp_confidence": round(mlp_prob * 100.0, 1) if mlp_prob is not None else None,
            "reference_landmarks": self.reference_landmarks.get(sid, []),
        }
