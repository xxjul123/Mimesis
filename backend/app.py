"""
app.py — Servidor web y API de análisis de MIMESIS (FastAPI).

¿Cómo funciona?
- Este archivo levanta un servidor HTTP local en http://localhost:8000.
- Sirve directamente tu interfaz web (index.html, game.html, css/, js/, assets/).
- Expone el endpoint POST /api/analysis/pose que recibe la foto desde JavaScript,
  ejecuta OpenCV (PDI) -> MediaPipe BlazePose -> Evaluador MLP y devuelve JSON.

Ejecución desde la raíz del proyecto:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .evaluator import PoseEvaluator
from .pdi import preprocess_for_pose
from .pose import PoseExtractor, extract_geometric_features

ROOT_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="MIMESIS API",
    description="Servicio de PDI (OpenCV) + MediaPipe BlazePose + Evaluador MLP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias únicas cargadas al iniciar el servidor
evaluator = PoseEvaluator()
pose_extractor: Optional[PoseExtractor] = None


def get_pose_extractor() -> PoseExtractor:
    global pose_extractor
    if pose_extractor is None:
        pose_extractor = PoseExtractor(model_complexity=1, min_detection_confidence=0.45)
    return pose_extractor


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mlp_loaded": evaluator.mlp_loaded,
        "sculptures": list(evaluator.references.keys()),
    }


@app.post("/api/analysis/pose")
async def analyze_pose(
    image: UploadFile = File(...),
    player: str = Form("Jugador"),
    sculpture_id: Optional[str] = Form(None),
    sculpture: Optional[str] = Form(None),
    round_num: str = Form("1", alias="round"),
):
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="La imagen enviada está vacía.")

    # Resolver ID de escultura (acepta sculpture_id directo o JSON en sculpture)
    sid = sculpture_id
    if not sid and sculpture:
        try:
            sid = json.loads(sculpture).get("id")
        except Exception:
            sid = None
    sid = (sid or "belalcazar").lower()

    # 1. Preprocesamiento PDI (OpenCV: Laplaciano + CLAHE en LAB + Letterboxing)
    try:
        boxed_rgb, meta, pdi_metrics = preprocess_for_pose(raw_bytes, target_size=640)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Estimación de pose (MediaPipe BlazePose)
    try:
        extracted = get_pose_extractor().extract(boxed_rgb, meta)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if extracted is None:
        raise HTTPException(
            status_code=422,
            detail="No se detectó una persona completa en la fotografía. Aléjate un poco de la cámara e intenta de nuevo.",
        )

    landmarks, mean_vis = extracted
    pdi_metrics["mean_visibility"] = round(mean_vis, 3)

    # 3. Normalización geométrica + Evaluación (MLP propio o modo geométrico)
    angles, _ = extract_geometric_features(landmarks)
    eval_result = evaluator.evaluate(angles, sid)

    blur_warning = (
        " (Aviso PDI: la foto salió algo borrosa; intenta quedarte quieto en el próximo intento)."
        if pdi_metrics["is_blurry"]
        else ""
    )

    return {
        "player": player,
        "sculpture_id": sid,
        "round": round_num,
        "score": eval_result["score"],
        "regions": eval_result["regions"],
        "feedback": eval_result["feedback"] + blur_warning,
        "evaluator_mode": eval_result["evaluator_mode"],
        "mlp_confidence": eval_result["mlp_confidence"],
        "landmarks": landmarks,
        "reference_landmarks": eval_result.get("reference_landmarks", []),
        "pdi_metrics": pdi_metrics,
        "message": f"Análisis completado para {player} ({eval_result['evaluator_mode']}).",
    }


# Servir el frontend estático desde el mismo servidor en http://localhost:8000
for folder in ("css", "js", "assets"):
    folder_path = ROOT_DIR / folder
    if folder_path.exists():
        app.mount(f"/{folder}", StaticFiles(directory=str(folder_path)), name=folder)


@app.get("/")
def serve_index():
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/index.html")
def serve_index_html():
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/game.html")
def serve_game_html():
    return FileResponse(ROOT_DIR / "game.html")
