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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .evaluator import PoseEvaluator
from .pdi import generate_pdi_debug_views, preprocess_with_intermediates
from .pose import PoseExtractor, extract_geometric_features, validate_body_framing
from .segmentation import refine_mask

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


@app.middleware("http")
async def disable_static_dev_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/index.html", "/game.html") or path.startswith(("/js/", "/css/")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# Instancias únicas cargadas al iniciar el servidor
evaluator = PoseEvaluator()
pose_extractor: Optional[PoseExtractor] = None


def get_pose_extractor() -> PoseExtractor:
    global pose_extractor
    if pose_extractor is None:
        pose_extractor = PoseExtractor(
            model_complexity=1,
            min_detection_confidence=0.30,
            enable_segmentation=True,
        )
    return pose_extractor


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mlp_loaded": evaluator.mlp_loaded,
        "sculptures": list(evaluator.references.keys()),
        "reference_masks_loaded": list(evaluator.reference_masks.keys()),
    }


@app.post("/api/analysis/pose")
async def analyze_pose(
    image: UploadFile = File(...),
    player: str = Form("Jugador"),
    sculpture_id: Optional[str] = Form(None),
    sculpture: Optional[str] = Form(None),
    round_num: str = Form("1", alias="round"),
    game_mode: str = Form("challenge"),
    include_debug_views: bool = Form(True),
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
    mode_norm = (game_mode or "challenge").lower()

    # 1. Preprocesamiento PDI (OpenCV: Laplaciano + CLAHE en LAB + Letterboxing)
    try:
        boxed_rgb, meta, pdi_metrics, img_bgr, enhanced_bgr = preprocess_with_intermediates(
            raw_bytes, target_size=640
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Estimación de pose + máscara de segmentación (MediaPipe BlazePose)
    try:
        extracted = get_pose_extractor().extract_with_mask(boxed_rgb, meta)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if extracted is None:
        raise HTTPException(
            status_code=422,
            detail="No se detectó una persona completa en la fotografía. Aléjate un poco de la cámara e intenta de nuevo.",
        )

    landmarks, mean_vis, raw_mask = extracted
    pdi_metrics["mean_visibility"] = round(mean_vis, 3)

    framing_ok, framing_msg = validate_body_framing(landmarks)
    if not framing_ok:
        raise HTTPException(status_code=422, detail=framing_msg)

    # 2B. Refinamiento morfológico de la máscara de segmentación (OpenCV)
    clean_mask = refine_mask(raw_mask, threshold=0.45) if raw_mask is not None else None
    pdi_metrics["segmentation_ok"] = clean_mask is not None

    # 3. Normalización geométrica + Clasificación / Evaluación + Similitud de silueta
    angles, _ = extract_geometric_features(landmarks)
    eval_result = evaluator.evaluate(
        angles,
        sculpture_id=None if mode_norm == "classify" else sid,
        game_mode=mode_norm,
        player_mask=clean_mask,
    )

    # 4. Vistas del Inspector Pipeline PDI + Modo Escultura (JPEG base64 livianas)
    debug_views = (
        generate_pdi_debug_views(img_bgr, enhanced_bgr, mask=clean_mask)
        if include_debug_views
        else None
    )

    blur_warning = (
        " (Aviso PDI: la foto salió algo borrosa; intenta quedarte quieto en el próximo intento)."
        if pdi_metrics["is_blurry"]
        else ""
    )

    resolved_sid = eval_result["sculpture_id"]
    return {
        "player": player,
        "game_mode": mode_norm,
        "sculpture_id": resolved_sid,
        "round": round_num,
        "score": eval_result["score"],
        "regions": eval_result["regions"],
        "feedback": eval_result["feedback"] + blur_warning,
        "evaluator_mode": eval_result["evaluator_mode"],
        "mlp_confidence": eval_result["mlp_confidence"],
        "predicted_class": eval_result["predicted_class"],
        "predicted_confidence": eval_result["predicted_confidence"],
        "is_recognized": eval_result["is_recognized"],
        "class_probabilities": eval_result["class_probabilities"],
        "silhouette_metrics": eval_result["silhouette_metrics"],
        "landmarks": landmarks,
        "reference_landmarks": eval_result.get("reference_landmarks", []),
        "pdi_metrics": pdi_metrics,
        "debug_views": debug_views,
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
