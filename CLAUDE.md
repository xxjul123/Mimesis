# CLAUDE.md — Contexto y Estado Actual de MIMESIS

Este documento resume todas las decisiones técnicas, los archivos creados/modificados, los problemas conocidos y los pasos pendientes del proyecto **MIMESIS** para continuar el trabajo sin perder contexto.

---

## 1. Contexto Académico y Decisión Arquitectónica (Camino A)

- **Materia:** Procesamiento Digital de Imágenes (UAO · Prof. Nicolás Llanos Neuta).
- **Rúbrica:** `Rubrica proyecto final PDI 2026-2.pdf` (temática: *Análisis de la Expresión Humana en el Arte y la Cultura*).
- **Especificación original:** `Desarrollo del sistema - Mimesis.pdf`.
- **Esculturas activas (3):** `belalcazar` (Sebastián de Belalcázar), `discobolo` (El Discóbolo) y `shakespeare` (William Shakespeare), alineadas con el dataset propio del equipo del Parcial 2 (480 fotos reales en Drive: 160 por clase, 128 train / 16 val / 16 test).

### Por qué elegimos el **Camino A**
- **NO se integran en producción** ni la CNN clasificadora del Parcial 2 (`Parcial2_Clasificacion_Poses_Esculturas.ipynb`) ni la U-Net de segmentación del Parcial 3 (`Modelo_Segmentacion_Parcial3.ipynb`), porque:
  - La CNN sobre píxeles RGB crudos se confunde con fondos, ropa e iluminación en escenas reales.
  - La U-Net (56 fotos de entrenamiento a 128×128) produce huecos internos y falsos positivos con fondos cargados, lo que empeoraría la detección de pose.
- **Cómo se aprovechan los parciales y las referencias:**
  1. **Pose de referencia exacta (ground truth geométrico):** se extrae **directamente de las 3 fotografías reales de las esculturas** (`assets/sculptures/*.jpg`) mediante `backend/extract_references.py` (BlazePose + refinamiento supervisado en `SUPERVISED_OVERRIDES`), guardándose en `backend/data/reference_poses.json`.
  2. **Rol del MLP con las 480 fotos del Parcial 2:** aunque los voluntarios no imitan la estatua al 100%, el MLP entrenado con los 12 ángulos de esas 480 fotos sirve como **verificador de clase/intención con variabilidad humana real** y permite presentar la **comparativa cuantitativa contra la CNN del Parcial 2 (83.3% val)** en el informe y póster.
  3. **Componente PDI medible:** OpenCV antes de la inferencia de pose (varianza del Laplaciano, CLAHE adaptativo en LAB y letterboxing sin deformar).

---

## 2. Arquitectura Actual Implementada

```text
Navegador (http://localhost:8000)          Backend Python (FastAPI · backend/app.py)
─────────────────────────────────          ─────────────────────────────────────────
index.html (Sala: 2–4 jugadores,           GET  /  /game.html  /css  /js  /assets
            3 esculturas reales)           GET  /api/health
game.html  (Selector cámara PC/USB,        POST /api/analysis/pose (multipart/form-data)
            rotar 0..270°, espejo,           ├─► 1. backend/pdi.py (OpenCV)
            modo vertical 3:4, 3s /          │      • Varianza del Laplaciano (nitidez)
            subir foto, canvas 15 pts        │      • CLAHE adaptativo en canal L (LAB)
            con colores + labels lado        │      • Letterboxing (640x640 sin deformar)
            a lado, mapa de calor,           ├─► 2. backend/pose.py (MediaPipe BlazePose)
            PDI y modal de ranking)          │      • 33 landmarks -> unletterbox a [0,1]
                                             │      • 12 ángulos invariantes (con fallback
                                             │        oreja visible -> nariz en perfil)
                                             └─► 3. backend/evaluator.py
                                                    • Similitud por región (head/torso/arms/legs)
                                                      evaluando normal + espejo (_mirror_angles)
                                                    • Devuelve también reference_landmarks
                                                    • PoseMLP (PyTorch) si existe pose_mlp.pt,
                                                      o modo geométrico de respaldo
```

---

## 3. Decisiones Técnicas Clave (Landmarks, Oclusiones y Espejo)

### 3.1 Los 15 landmarks de calificación vs. 33 de BlazePose
- BlazePose siempre devuelve **33 landmarks (`0..32`)** y todos se siguen guardando en `reference_poses.json`.
- Sin embargo, la evaluación angular (`backend/pose.py`) usa únicamente **15 landmarks clave**, y **solo esos 15 se dibujan** tanto en las previsualizaciones de Python (`backend/extract_references.py`) como en el `<canvas>` de la interfaz web (`js/game.js`):
  - **Cabeza** (amarillo `#ffd71e`): `0:Nariz`, `7:Oreja I`, `8:Oreja D`
  - **Torso** (blanco/plata `#ebebeb`): conexiones `(11,12)`, `(11,23)`, `(12,24)`, `(23,24)`
  - **Brazo Izq** (cian `#32d7ff`): `11:Hombro I`, `13:Codo I`, `15:Muñeca I`
  - **Brazo Der** (naranja `#ff6e41`): `12:Hombro D`, `14:Codo D`, `16:Muñeca D`
  - **Pierna Izq** (verde `#73eb5f`): `23:Cadera I`, `25:Rodilla I`, `27:Tobillo I`
  - **Pierna Der** (magenta `#dc64eb`): `24:Cadera D`, `26:Rodilla D`, `28:Tobillo D`
- Se excluyen visualmente ojos (`1..6`), boca (`9..10`), dedos (`17..22`) y pies/talones (`29..32`).
- Cada punto lleva etiqueta (`índice:Nombre`) con posicionamiento anti-solapamiento, línea guía y leyenda de segmentos en la esquina superior izquierda.

### 3.2 Manejo de oreja oculta en vistas de perfil / 3/4 (`_compute_head_tilt`)
- En las 3 esculturas la cabeza está girada hacia su izquierda, por lo que `7:Oreja I` queda oculta detrás de la cabeza y BlazePose la estimaba de forma errática sobre el ojo o la mejilla.
- **Solución implementada en `backend/pose.py` (`_compute_head_tilt`):**
  - **Vista frontal** (ambas orejas `7` y `8` con `visibility >= 0.45` y separación horizontal `|x_7 - x_8| >= 0.20 * ancho_hombros`): usa la línea clásica `Oreja I (7) ↔ Oreja D (8)`.
  - **Vista de perfil / 3/4** (una oreja oculta o comprimida horizontalmente): usa la línea **`Oreja visible (7 u 8) → Nariz (0)`**.
  - Mantiene intacto el vector de **12 características (`FEATURE_ORDER`)** para conservar compatibilidad total con `PoseMLP`.
- En `backend/extract_references.py`, `SUPERVISED_OVERRIDES` soporta tuplas `(x, y)` (visibilidad `0.99`) o `(x, y, visibility)`; en las 3 esculturas `7:Oreja I` está marcada con `visibility = 0.0` para que no se dibuje ni interfiera en `head_tilt`.

### 3.3 Simetría izquierda/derecha (imitación anatómica vs. en espejo)
- **Totalmente soportada:**
  - En `backend/evaluator.py`, `evaluate()` y `_mlp_probability()` evalúan cada foto **dos veces**: con los ángulos originales y con `_mirror_angles()` (intercambiando `left_* ↔ right_*` en codos, hombros, caderas y rodillas), conservando automáticamente la orientación de mayor similitud.
  - `head_tilt`, `neck_angle`, `torso_lean` y `shoulder_level` son magnitudes sin signo (`0..180°` / `abs`), por lo que ya son invariantes a espejo horizontal.
  - En el frontend (`js/game.js`), el usuario puede activar/desactivar el efecto espejo (`Espejo: Sí/No`) según use cámara frontal (selfie) o trasera.

---

## 4. Mapa de Archivos Creados / Modificados

### Frontend (estética editorial de museo oscuro cálido)
- `assets/sculptures/{belalcazar,discobolo,shakespeare}.jpg` — **Fotografías reales** de las 3 esculturas (ya cargadas).
- `css/styles.css` — Paleta carbón cálido (`#12110f`), marfil (`#f3eee6`), acento bronce/terracota (`#c88d55`), fuentes *Cormorant Garamond* + *DM Sans*, colores semánticos por puntaje.
- `css/layout.css` — Grids responsive para sala y partida.
- `css/components.css` — Fichas museográficas, barra de selección de cámara (`.camera-device-bar`, `.camera-select`), soporte vertical `.camera-frame.portrait` (`3:4`, `object-fit: contain`), mapa de calor, diagnóstico PDI y modal de Ranking.
- `js/sculptures.js` — Catálogo centralizado (`window.MIMESIS_SCULPTURES`) con las 3 esculturas.
- `js/lobby.js` — Gestión de 2–4 jugadores, selección de obra, verificación de `/api/health` y persistencia en `sessionStorage`.
- `js/game.js` —
  - **Cámara multi-dispositivo y vertical:** selector de cámara (`enumerateDevices` + evento `devicechange` para webcam de PC o celular por USB), cambio en vivo, rotación `0°/90°/180°/270°`, toggle de espejo, adaptación automática a marco vertical (`portrait`) y captura en `<canvas>` respetando rotación y espejo. No fuerza `1280×720` al elegir un `deviceId` específico para mejorar compatibilidad con cámaras virtuales.
  - **Visualización de pose en `<canvas>` (`drawPoseOverlay`):** reemplazó el SVG original (que se desfasaba en fotos verticales). Dibuja lado a lado el esqueleto del jugador y el esqueleto real de la escultura (`reference_landmarks`), usando solo los **15 landmarks de calificación**, 6 colores por segmento, leyenda y etiquetas anti-solapamiento.
  - Barras de mapa de calor, diagnóstico PDI y modal de Ranking con rotación automática entre esculturas.
- `index.html` y `game.html` — HTML semántico con barra de dispositivos de cámara y doble etapa de comparación (`#player-result-stage` y `#reference-result-stage`).

### Backend Python (`backend/`)
- `backend/requirements.txt` — `fastapi`, `uvicorn[standard]`, `python-multipart`, `opencv-python`, `mediapipe>=0.10.30`, `numpy`, `torch`.
- `backend/pdi.py` — `decode_image`, `compute_sharpness` (Laplaciano, umbral `40.0`), `apply_adaptive_clahe` (LAB canal `L`, `clipLimit=2.0`, `tileGridSize=(8,8)`), `letterbox_resize` (`640x640`), `unletterbox_point`, `preprocess_for_pose`.
- `backend/pose.py` — `PoseExtractor` compatible con **MediaPipe 1.x (`mp.tasks.vision.PoseLandmarker`)** y clásico (`mp.solutions.pose`). Descarga automáticamente `backend/models/pose_landmarker_full.task` y **lo carga con `model_asset_buffer=model_path.read_bytes()`** (crítico en Windows por espacios en la ruta). Incluye `_compute_head_tilt` con fallback `oreja visible → nariz` en perfil.
- `backend/extract_references.py` — Script offline (`python -m backend.extract_references`) que:
  - Extrae los 33 landmarks de las 3 fotos reales (usando recorte ROI para `belalcazar.jpg` para separarlo del fondo de árboles).
  - Aplica `SUPERVISED_OVERRIDES` verificados visualmente:
    - `belalcazar`: nariz (`0`), oreja D (`8`), oreja I oculta (`7`, vis `0.0`), brazos y piernas.
    - `discobolo`: nariz (`0`), oreja D (`8`), oreja I oculta (`7`, vis `0.0`), brazos, caderas (`23`/`24`) y rodillas (`25`/`26`) separando correctamente pierna delantera apoyada (derecha) y trasera (izquierda).
    - `shakespeare`: nariz (`0`), oreja D (`8`), oreja I oculta (`7`, vis `0.0`), codo derecho (`14`) y muñeca derecha (`16`) reubicada detrás de la cintura.
  - Genera `backend/data/reference_poses.json` (33 landmarks + 12 ángulos reales) y previsualizaciones coloreadas con los 15 puntos y etiquetas en `backend/data/reference_previews/*_skeleton.jpg`.
- `backend/data/reference_poses.json` — Referencias reales (landmarks + ángulos) cargadas por `PoseEvaluator`.
- `backend/evaluator.py` — `PoseMLP` (PyTorch) y `PoseEvaluator`: evalúa normal + espejo (`_mirror_angles`), calcula similitud por región (`head`, `torso`, `arms`, `legs`), combina `0.60 * mlp_prob + 0.40 * geo_score` cuando existe `backend/models/pose_mlp.pt` (si no, modo `geometric`) y retorna `reference_landmarks`.
- `backend/app.py` — Servidor FastAPI (`GET /api/health`, `POST /api/analysis/pose` devolviendo `landmarks`, `reference_landmarks`, `pdi_metrics`, `regions`, `score`, `feedback`).

---

## 5. Contrato Exacto para Entrenar `PoseMLP` (Compatibilidad con `backend/evaluator.py`)

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

> **Nota para Colab:** replicar la misma lógica de `_compute_head_tilt` (fallback `oreja visible → nariz` en perfil) al extraer los 12 ángulos de las 480 fotos para que la distribución de entrenamiento coincida exactamente con producción (`backend/pose.py`).

---

## 6. Entorno Local Verificado y Notas Conocidas

- **SO:** Windows · **Python:** 3.13 (instalado desde Microsoft Store; ejecutar módulos con `python -m ...`).
- **Regenerar referencias y previsualizaciones tras editar `SUPERVISED_OVERRIDES`:**
  ```powershell
  python -m backend.extract_references
  ```
- **Levantar el servidor local:**
  ```powershell
  python -m uvicorn backend.app:app --reload
  ```
- **URL local:** `http://localhost:8000`
- **Notas prácticas observadas en pruebas:**
  - En `backend/app.py`, `min_detection_confidence=0.45`: si el jugador sale cortado en la cintura (sin piernas) o con contraluz fuerte, BlazePose puede devolver HTTP 422 (`"No se detectó una persona completa..."`). Conviene alejarse para encuadre de cuerpo completo (o bajar el umbral a `~0.25` si se requiere mayor tolerancia en webcam).
  - El mensaje de error en `js/game.js` (`processCapture`) actualmente añade `"Verifica que el servidor Python esté ejecutándose..."` incluso ante respuestas HTTP 422 válidas del servidor (pendiente refinar para distinguir error de red vs. error de validación).
  - En celulares Samsung (ej. Galaxy A26 con One UI) no aparece el modo nativo "Cámara web" por USB; con DroidCam Client v7+ en Chrome puede aparecer pantalla verde por conflicto del driver virtual (alternativas: reiniciar DroidCam antes de abrir Chrome, activar acceso multi-app en Windows, usar **Iriun Webcam** por USB o probar en Firefox).

---

## 7. Tareas Pendientes (Roadmap)

1. **Verificación final de referencias regeneradas:** Correr `python -m backend.extract_references` tras los últimos ajustes de cabeza/Discóbolo/Shakespeare y confirmar visualmente `backend/data/reference_previews/*_skeleton.jpg`.
2. **Ajustes menores de UX/detección (opcionales):**
   - Evaluar bajar `min_detection_confidence` en `backend/app.py` (ej. `0.25`–`0.30`) para webcams con contraluz.
   - Separar en `js/game.js` errores de red (servidor apagado) vs. errores HTTP 422 del backend (sin persona detectada).
3. **Notebook de Colab (`notebooks/Mimesis_MLP_Entrenamiento.ipynb`):**
   - Leer las 480 fotos de Drive (`PDI - Parcial2`, respetando `train / val / test` sin fuga de datos).
   - **Experimento PDI:** medir tasa de detección y visibilidad promedio de BlazePose **sin CLAHE vs. con CLAHE + letterboxing**.
   - Entrenar `PoseMLP` (con aumento en espejo izquierdo/derecho y usando `_compute_head_tilt`), evaluar en `test` (Accuracy, F1, matriz de confusión) y generar la **tabla comparativa contra la CNN del Parcial 2 (83.3% val)**.
   - Exportar `pose_mlp.pt` y copiarlo a `backend/models/`.
4. **Pruebas unitarias (`backend/tests/test_pipeline.py`):** Validar Laplaciano, CLAHE, letterboxing + unletterbox, `_compute_head_tilt` (frontal vs. perfil), invarianza en espejo y esquema JSON del endpoint.
5. **Actualizar `README.md`:** Reemplazar el contenido actual (habla por error de un "Juego de Simulación 3D") por la documentación oficial de MIMESIS con las 8 secciones exigidas en `Rubrica proyecto final PDI 2026-2.pdf`.
6. **Entregables académicos finales:** Póster académico (Figura 1 de la rúbrica), video demostrativo (máx. 3 min) y sustentación.
