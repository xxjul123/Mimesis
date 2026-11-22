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

import cv2
import numpy as np

from .pose import FEATURE_ORDER, REGION_ANGLES
from .segmentation import compare_silhouettes

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

DATA_PATH = Path(__file__).parent / "data" / "reference_poses.json"
MASKS_DIR = Path(__file__).parent / "data" / "reference_masks"
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
        self.reference_masks: Dict[str, np.ndarray] = {}
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

            mask_path = MASKS_DIR / f"{k}.png"
            if mask_path.exists():
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    self.reference_masks[k] = mask_img

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

    def _best_regions_for(self, angles: Dict[str, float], sid: str) -> Tuple[Dict[str, float], float]:
        ref = self.references.get(sid) or next(iter(self.references.values()))
        reg_normal = _region_scores(angles, ref)
        reg_mirror = _region_scores(_mirror_angles(angles), ref)
        tot_normal = _weighted_total(reg_normal)
        tot_mirror = _weighted_total(reg_mirror)
        if tot_mirror > tot_normal:
            return reg_mirror, tot_mirror
        return reg_normal, tot_normal

    def _mlp_all_probabilities(
        self,
        angles: Dict[str, float],
        focus_sid: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Ejecuta PoseMLP sobre la pose normal y su versión espejo, devolviendo
        las probabilidades (0..1) de las 3 esculturas.
        """
        if not self.mlp_loaded or self.mlp is None or torch is None:
            return None

        vec_normal = np.array([angles[k] / 180.0 for k in FEATURE_ORDER], dtype=np.float32)
        mirrored = _mirror_angles(angles)
        vec_mirror = np.array([mirrored[k] / 180.0 for k in FEATURE_ORDER], dtype=np.float32)

        with torch.no_grad():
            x = torch.from_numpy(np.stack([vec_normal, vec_mirror], axis=0))
            probs = torch.softmax(self.mlp(x), dim=1).numpy()  # shape (2, num_classes)

        if focus_sid and focus_sid in SCULPTURE_CLASSES:
            idx = SCULPTURE_CLASSES.index(focus_sid)
            chosen = probs[1] if probs[1, idx] > probs[0, idx] else probs[0]
        else:
            chosen = probs[1] if float(np.max(probs[1])) > float(np.max(probs[0])) else probs[0]

        return {cls: float(chosen[i]) for i, cls in enumerate(SCULPTURE_CLASSES)}

    def _geometric_class_probabilities(self, geo_by_class: Dict[str, float]) -> Dict[str, float]:
        """Convierte los puntajes geométricos por clase en una distribución softmax (0..1)."""
        scores = np.array([geo_by_class.get(cls, 0.0) for cls in SCULPTURE_CLASSES], dtype=np.float64)
        scaled = (scores - np.max(scores)) / 10.0
        exp_s = np.exp(scaled)
        probs = exp_s / max(float(np.sum(exp_s)), 1e-9)
        return {cls: float(probs[i]) for i, cls in enumerate(SCULPTURE_CLASSES)}

    def evaluate(
        self,
        angles: Dict[str, float],
        sculpture_id: Optional[str] = None,
        game_mode: str = "challenge",
        player_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        mode_norm = (game_mode or "challenge").lower()
        is_classify = mode_norm == "classify"

        # Calcular similitud geométrica contra las 3 esculturas
        regions_by_class: Dict[str, Dict[str, float]] = {}
        geo_by_class: Dict[str, float] = {}
        for cls in SCULPTURE_CLASSES:
            if cls in self.references:
                regs, tot = self._best_regions_for(angles, cls)
                regions_by_class[cls] = regs
                geo_by_class[cls] = tot

        focus_sid = None if is_classify else (sculpture_id or "belalcazar").lower()
        mlp_probs = self._mlp_all_probabilities(angles, focus_sid=focus_sid)

        if mlp_probs is not None:
            raw_probs = mlp_probs
            eval_mode = "mlp"
        else:
            raw_probs = self._geometric_class_probabilities(geo_by_class)
            eval_mode = "geometric"

        predicted_class = max(raw_probs, key=lambda c: raw_probs[c])
        predicted_conf = raw_probs[predicted_class] * 100.0

        # En Modo Libre se evalúa contra la escultura predicha; en Modo Reto contra la asignada
        target_sid = predicted_class if is_classify else (sculpture_id or "belalcazar").lower()
        if target_sid not in regions_by_class:
            target_sid = next(iter(regions_by_class.keys()))

        regions = regions_by_class[target_sid]
        geo_score = geo_by_class[target_sid]
        target_mlp_prob = mlp_probs.get(target_sid) if mlp_probs is not None else None

        if target_mlp_prob is not None:
            final_score = 0.60 * (target_mlp_prob * 100.0) + 0.40 * geo_score
        else:
            final_score = geo_score

        final_score = round(float(np.clip(final_score, 0.0, 100.0)), 1)

        # Detección de postura fuera de distribución (Modo Libre)
        is_recognized = True
        if is_classify and predicted_conf < 45.0 and geo_score < 45.0:
            is_recognized = False
            feedback = (
                "Postura no reconocida: tu pose no coincide claramente con ninguna de las 3 esculturas. "
                "Elige una obra de referencia e intenta imitarla de cuerpo completo."
            )
        else:
            feedback = _build_feedback(regions, final_score)

        # Métrica diagnóstica de silueta (IoU + Momentos de Hu) sin alterar la fórmula principal
        ref_mask = self.reference_masks.get(target_sid)
        silhouette_metrics = compare_silhouettes(player_mask, ref_mask)

        class_probabilities = {cls: round(raw_probs[cls] * 100.0, 1) for cls in SCULPTURE_CLASSES}

        return {
            "sculpture_id": target_sid,
            "score": final_score,
            "regions": regions,
            "feedback": feedback,
            "evaluator_mode": eval_mode,
            "mlp_confidence": round(target_mlp_prob * 100.0, 1) if target_mlp_prob is not None else None,
            "predicted_class": predicted_class,
            "predicted_confidence": round(predicted_conf, 1),
            "is_recognized": is_recognized,
            "class_probabilities": class_probabilities,
            "silhouette_metrics": silhouette_metrics,
            "reference_landmarks": self.reference_landmarks.get(target_sid, []),
        }
