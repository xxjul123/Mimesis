# CLAUDE.md — Contexto y Estado Actual de MIMESIS

Este documento resume todas las decisiones técnicas, los archivos creados/modificados y los pasos pendientes del proyecto **MIMESIS** para que puedas continuar el trabajo sin perder contexto.

---

## 1. Contexto Académico y Decisión Arquitectónica (Camino A)

- **Materia:** Procesamiento Digital de Imágenes (UAO · Prof. Nicolás Llanos Neuta).
- **Rúbrica:** `Rubrica proyecto final PDI 2026-2.pdf` (temática: *Análisis de la Expresión Humana en el Arte y la Cultura*).
- **Especificación original:** `Desarrollo del sistema - Mimesis.pdf`.
- **Esculturas activas (solo 3 por ahora):** `belalcazar` (Sebastián de Belalcázar), `discobolo` (El Discóbolo) y `shakespeare` (William Shakespeare), alineadas con el dataset propio del equipo del Parcial 2 (480 fotos reales en Drive: 160 por clase, 128 train / 16 val / 16 test).

### Por qué elegimos el **Camino A**
- **NO se integran en producción** ni la CNN clasificadora del Parcial 2 (`Parcial2_Clasificacion_Poses_Esculturas.ipynb`) ni la U-Net de segmentación del Parcial 3 (`Modelo_Segmentacion_Parcial3.ipynb`), porque:
  - La CNN sobre píxeles RGB crudos se confunde con fondos, ropa e iluminación en escenas reales.
  - La U-Net (56 fotos de entrenamiento a 128×128) produce huecos internos y falsos positivos con fondos cargados, lo que empeoraría la detección de pose.
- **SÍ se aprovechan los parciales así:**
  1. El Parcial 2 se usa como **baseline comparativo** en documentación y sustentación para justificar técnicamente el paso a keypoints normalizados + MLP.
  2. Se reutilizan las **480 fotos reales del Parcial 2** para extraer landmarks con BlazePose, construir `reference_poses.json` y entrenar el MLP propio (`pose_mlp.pt`).
  3. El componente de **Procesamiento Digital de Imágenes (PDI)** se implementa de forma real y medible con OpenCV antes de la inferencia de pose.

---

## 2. Arquitectura Actual Implementada

```text
Navegador (http://localhost:8000)          Backend Python (FastAPI · backend/app.py)
─────────────────────────────────          ─────────────────────────────────────────
index.html (Sala: 2–4 jugadores,           GET  /  /game.html  /css  /js  /assets
            3 esculturas)                  GET  /api/health
game.html  (Cámara 3s / subir foto,        POST /api/analysis/pose (multipart/form-data)
            esqueleto por zona,              ├─► 1. backend/pdi.py (OpenCV)
            mapa de calor, PDI,              │      • Varianza del Laplaciano (nitidez)
            modal de ranking)                │      • CLAHE adaptativo en canal L (LAB)
                                             │      • Letterboxing (640x640 sin deformar)
                                             ├─► 2. backend/pose.py (MediaPipe BlazePose)
                                             │      • 33 landmarks -> unletterbox a [0,1]
                                             │      • 12 ángulos corporales invariantes
                                             └─► 3. backend/evaluator.py
                                                    • Similitud por región (head/torso/arms/legs)
                                                    • PoseMLP (PyTorch) si existe pose_mlp.pt,
                                                      o modo geométrico de respaldo
```

---

## 3. Mapa de Archivos Creados / Modificados

### Frontend (rediseñado con estética editorial de museo oscuro cálido)
Se eliminó el estilo anterior (azul oscuro + neón cian + símbolos `◈/◉` + CSS/JS inline + 5 obras genéricas de Unsplash) y se separó limpiamente en módulos:

- `css/styles.css` — Paleta carbón cálido (`#12110f`), marfil (`#f3eee6`), acento bronce/terracota (`#c88d55`), fuentes *Cormorant Garamond* + *DM Sans*, colores semánticos por puntaje (`--score-high`, `--score-mid`, `--score-low`).
- `css/layout.css` — Grids responsive para sala (3 columnas de galería) y partida.
- `css/components.css` — Fichas museográficas, cámara, mapa de calor, tarjetas de diagnóstico PDI y modal de Ranking.
- `js/sculptures.js` — Catálogo centralizado (`window.MIMESIS_SCULPTURES`) con las 3 esculturas (`belalcazar`, `discobolo`, `shakespeare`). Usa `assets/sculptures/<id>.jpg` y cae automáticamente a `assets/sculptures/<id>.svg` si aún no se copian las fotos reales.
- `js/lobby.js` — Gestión de 2–4 jugadores con edición inline de nombres, selección de obra, verificación de `/api/health` y persistencia en `sessionStorage`.
- `js/game.js` — Cámara web con cuenta regresiva (3s), botón **"Subir foto de prueba"**, envío a `POST /api/analysis/pose`, superposición de esqueleto SVG coloreado por región corporal, barras de mapa de calor, tarjeta de diagnóstico PDI y **modal de Ranking** al cierre de cada ronda (con rotación automática entre las 3 esculturas).
- `index.html` y `game.html` — HTML semántico limpio con rutas relativas (`css/...`, `js/...`).
- `assets/sculptures/{belalcazar,discobolo,shakespeare}.svg` — Placas SVG minimalistas temporales.

### Backend Python (`backend/`)
- `backend/requirements.txt` — `fastapi`, `uvicorn[standard]`, `python-multipart`, `opencv-python`, `mediapipe>=0.10.30`, `numpy`, `torch`.
- `backend/pdi.py` — `decode_image`, `compute_sharpness` (Laplaciano, umbral `40.0`), `apply_adaptive_clahe` (LAB canal `L`, `clipLimit=2.0`, `tileGridSize=(8,8)`), `letterbox_resize` (`640x640`), `unletterbox_point`, `preprocess_for_pose`.
- `backend/pose.py` — `PoseExtractor` compatible con **MediaPipe 1.x (`mp.tasks.vision.PoseLandmarker`)** y clásico (`mp.solutions.pose`). Descarga automáticamente `backend/models/pose_landmarker_full.task` si no existe y **lo carga con `model_asset_buffer=model_path.read_bytes()`** (crítico en Windows porque la ruta del proyecto contiene espacios y el cargador C++ de MediaPipe falla con `model_asset_path`).
- `backend/evaluator.py` — `PoseMLP` (PyTorch) y `PoseEvaluator`: evalúa en orientación normal y espejo (left/right), calcula similitud por región (`head`, `torso`, `arms`, `legs`) y combina `0.60 * mlp_prob + 0.40 * geo_score` cuando `backend/models/pose_mlp.pt` existe (si no, funciona en modo `geometric`).
- `backend/data/reference_poses.json` — Ángulos iniciales aproximados (en grados) para las 3 esculturas; deben sobrescribirse con los prototipos reales extraídos del dataset en Colab.
- `backend/app.py` — Servidor FastAPI que sirve el frontend en `http://localhost:8000` y expone `GET /api/health` y `POST /api/analysis/pose`.

---

## 4. Contrato Exacto para Entrenar `PoseMLP` (Compatibilidad con `backend/evaluator.py`)

Cualquier notebook de entrenamiento (Colab) debe respetar exactamente este orden de 12 características (ángulos en grados divididos entre `180.0` → rango `[0, 1]`):

```python
FEATURE_ORDER = [
    "head_tilt", "neck_angle",
    "torso_lean", "shoulder_level",
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee",
]
SCULPTURE_CLASSES = ["belalcazar", "discobolo", "shakespeare"]
```

Arquitectura esperada en `backend/models/pose_mlp.pt` (definida en `backend/evaluator.py`):
```python
class PoseMLP(nn.Module):
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
```
El checkpoint puede guardarse como `torch.save({"state_dict": model.state_dict()}, "pose_mlp.pt")` o directamente `model.state_dict()`.

---

## 5. Entorno Local Verificado

- **SO:** Windows · **Python:** 3.13 (instalado desde Microsoft Store; `uvicorn` se ejecuta como módulo para evitar problemas de PATH).
- **Comando para levantar el servidor local:**
  ```powershell
  python -m uvicorn backend.app:app --reload
  ```
- **URL local:** `http://localhost:8000`
- **Estado probado:** El flujo completo (`index.html` → `game.html` → cámara / foto → `POST /api/analysis/pose` → PDI + BlazePose + modo geométrico → esqueleto + puntaje + métricas PDI) **ya fue probado y funciona en local**.

---

## 6. Tareas Pendientes (Roadmap)

1. **Imágenes reales de las 3 esculturas:** Colocar `belalcazar.jpg`, `discobolo.jpg` y `shakespeare.jpg` en `assets/sculptures/` (el frontend las tomará automáticamente en lugar de los `.svg`).
2. **Notebook de Colab (`notebooks/Mimesis_MLP_Entrenamiento.ipynb`):**
   - Leer las 480 fotos de Drive (`PDI - Parcial2`, respetando `train / val / test` sin fuga de datos).
   - **Experimento PDI:** medir tasa de detección y visibilidad promedio de BlazePose **sin CLAHE vs. con CLAHE + letterboxing**.
   - Calcular la mediana/media de los 12 ángulos por clase y exportar `reference_poses.json`.
   - Entrenar `PoseMLP` (con aumento en espejo izquierdo/derecho), evaluar en `test` (Accuracy, F1, matriz de confusión) y generar la **tabla comparativa contra la CNN del Parcial 2 (83.3 % val)**.
   - Exportar `pose_mlp.pt` y `reference_poses.json` para copiarlos a `backend/models/` y `backend/data/`.
3. **Pruebas unitarias (`backend/tests/test_pipeline.py`):** Validar Laplaciano, CLAHE, letterboxing + unletterbox, invarianza de ángulos y esquema JSON del endpoint.
4. **Actualizar `README.md`:** Reemplazar el contenido actual (habla por error de un "Juego de Simulación 3D") por la documentación oficial de MIMESIS con las 8 secciones exigidas en `Rubrica proyecto final PDI 2026-2.pdf`.
5. **Entregables académicos finales:** Póster académico (Figura 1 de la rúbrica), video demostrativo (máx. 3 min) y sustentación.
