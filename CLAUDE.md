# CLAUDE.md — Contexto y Estado Actual de MIMESIS

Documento de referencia actualizado con el estado vigente, la arquitectura implementada, los cambios recientes y los pendientes reales del proyecto **MIMESIS** para continuar sin perder contexto.

---

## 1. Contexto Académico y Enfoque Técnico (Camino A)

- **Materia:** Procesamiento Digital de Imágenes (UAO · Prof. Nicolás Llanos Neuta · 2026-2).
- **Rúbrica:** `Rubrica proyecto final PDI 2026-2.pdf` (*Análisis de la Expresión Humana en el Arte y la Cultura*).
- **Especificación inicial:** `Desarrollo del sistema - Mimesis.pdf` (v1.0; la arquitectura definitiva implementada es Web + FastAPI + PyTorch).
- **Esculturas activas (3):** `belalcazar` (Sebastián de Belalcázar), `discobolo` (El Discóbolo) y `shakespeare` (William Shakespeare).
- **Dataset real depurado (329 fotos en Google Drive · carpeta `PDI - Parcial2`, ignorando `sobrantes`):**
  - `train`: `258` imágenes (`86 × 3`; `514` muestras tras aumento en espejo horizontal aplicado únicamente sobre `train`).
  - `val`: `23` imágenes (`9` Belalcázar · `5` Discóbolo · `9` Shakespeare).
  - `test`: `48` imágenes balanceadas (`16 × 3`).
- **Decisión arquitectónica:** En producción no se usan la CNN RGB del Parcial 2 ni la U-Net del Parcial 3 (sensibles a fondo, ropa y baja resolución). Se integra un pipeline completo y medible de **PDI clásico (OpenCV) + Detección de pose y segmentación (MediaPipe `PoseLandmarker`) + Morfología/Silueta (`OpenCV`) + Clasificador propio (`PoseMLP` en PyTorch)** entrenado sobre 12 ángulos corporales invariantes.

---

## 2. Arquitectura Vigente

```text
Navegador (http://localhost:8000)                Backend Python (FastAPI · backend/app.py)
─────────────────────────────────                ─────────────────────────────────────────
index.html (Sala: 2–4 jugadores,                 GET  /  /game.html  /css  /js  /assets
            Modo Reto vs. Modo Libre,                 (middleware Cache-Control: no-cache en dev)
            3 esculturas reales)                 GET  /api/health
game.html  • Modo Manual: cámara PC/USB,         POST /api/analysis/pose (multipart/form-data)
             rotar 0..270°, espejo, 3s /           ├─► 1. backend/pdi.py (OpenCV)
             subir foto, 3 vistas (Segmentos,      │      • Laplaciano (nitidez, umbral 40.0)
             Mapa de calor, Escultura),            │      • CLAHE adaptativo en canal L (LAB)
             barras ML, Inspector PDI +            │      • Letterboxing 640×640 + unletterbox
             Lightbox ampliable, Ranking.          │      • Vistas debug (max_side=960)
           • Modo Automático Pantalla Completa     ├─► 2. backend/pose.py (MediaPipe PoseLandmarker)
             (auto_mode.js): 3 obras seguidas,     │      • 33 landmarks + máscara de segmentación
             10s pose (nombre arriba, obra en      │      • validate_body_framing (exige cadera+rodilla)
             esquina), 5s cambio, 8s resultados    │      • 12 ángulos invariantes + _compute_head_tilt
             pausables, Informe Detallado con      ├─► 3. backend/segmentation.py (OpenCV)
             congelamiento de reloj, Lightbox      │      • refine_mask (morfología + componente mayor)
             PDI, Web Audio y Podio Final.         │      • findContours + IoU alineado + Momentos de Hu
                                                   └─► 4. backend/evaluator.py (PoseMLP + Geométrico)
                                                          • Normal + espejo (_mirror_angles)
                                                          • Compuerta geométrica + rechazo OOD
                                                            (neutra, T-Pose/brazos arriba, geo < 50%)
                                                          • Modo Reto y Modo Libre ("Adivina la Obra")
```

---

## 3. Decisiones Técnicas Vigentes

### 3.1 Landmarks, Oclusiones y Encuadre Corporal
- BlazePose extrae 33 landmarks, pero la calificación y el dibujo `<canvas>` (`drawPoseOverlay`) usan los **15 landmarks clave** en 6 segmentos con colores fijos, leyenda y etiquetas anti-solapamiento (`0, 7, 8, 11–16, 23–28`).
- **`_compute_head_tilt` (`backend/pose.py`):** usa `Oreja I ↔ Oreja D` en vista frontal y hace fallback a `Oreja visible → Nariz` en perfil/3/4 cuando una oreja queda oculta (`visibility < 0.45` o compresión horizontal).
- **`validate_body_framing` (`backend/pose.py` + `backend/app.py`):** exige cabeza, hombros, al menos un brazo, al menos una cadera y **al menos una rodilla** dentro del encuadre (`visibility >= 0.35`, `y <= 1.03`). Rechaza con `HTTP 422` planos cerrados/sentados donde BlazePose inventa piernas fuera de vista, tolerando tobillos en el borde inferior en webcams de escritorio.
- **`min_detection_confidence = 0.30`** en `backend/app.py` con `enable_segmentation = True`.

### 3.2 Segmentación Morfológica y Diagnóstico de Forma (`backend/segmentation.py`)
- La máscara de segmentación de BlazePose se proyecta con `unletterbox_mask` y se refina con apertura/cierre morfológico elíptico (`cv2.morphologyEx`) + componente conexo principal (`connectedComponentsWithStats`) + contorno externo (`cv2.findContours`).
- **`compare_silhouettes`:** normaliza siluetas a `128×128` preservando relación de aspecto (normal y espejo) y calcula **IoU** y **Momentos de Hu** (`cv2.matchShapes`) contra las máscaras de referencia en `backend/data/reference_masks/*.png`.
- IoU y Hu se reportan como **métricas diagnósticas** (fuera de la fórmula del puntaje) porque la ropa, accesorios de las estatuas y encuadre afectan la silueta binaria más que los 12 ángulos articulares.

### 3.3 Clasificación `PoseMLP`, Simetría en Espejo y Rechazo OOD (`backend/evaluator.py`)
- **Modelo activo en producción:** `backend/models/pose_mlp.pt` (`17.7 KB`), arquitectura `12 → 64 (BN+ReLU+Dropout 0.25) → 32 (ReLU+Dropout 0.15) → 3`, entrenado sobre `FEATURE_ORDER` (12 ángulos divididos por `180.0`) y `SCULPTURE_CLASSES = ["belalcazar", "discobolo", "shakespeare"]`.
- **Invarianza a espejo:** tanto las regiones angulares como `PoseMLP` se evalúan en orientación normal y espejo (`_mirror_angles`), eligiendo la de mayor coincidencia.
- **Compuerta geométrica y rechazo OOD (*anti-overconfidence* de `softmax`):**
  - `_is_out_of_distribution_pose(angles)` detecta posturas ajenas a las 3 obras: postura neutra de pie (`max_shoulder < 20°`, `min_elbow > 150°`, `torso_lean < 12°`, `min_knee > 160°`) y posturas erguidas con ambos brazos levantados o en cruz (`min_shoulder >= 65°` y `torso_lean < 20°`, ej. T-Pose, Y-Pose, manos arriba).
  - **Compuerta geométrica:** `geo_gate = 0.0` si es OOD, si no `clip((geo_score - 25) / 35, 0, 1)`. Mantiene 100% del peso MLP (`0.60 * mlp + 0.40 * geo`) en imitaciones reales (`geo >= 60%`) y evita que un `softmax` inflado regale puntaje alto cuando la geometría real es baja.
  - **Modo Libre (`classify`):** marca `is_recognized = False` y `score = 0.0` (*"Postura no reconocida"*) si `geo_score < 50%`, si `predicted_conf < 55% and geo_score < 60%`, o si es postura OOD.
  - `_build_feedback` ajusta concordancia singular/plural (*"el torso coincide"* vs. *"los brazos coinciden"*).

### 3.4 Modo Automático en Pantalla Completa, Informe Detallado y Lightbox PDI
- **Modo manual preservado** en `game.html` + botón **«⛶ Pantalla completa automática»** (`#open-auto-mode-btn`).
- **Módulo independiente `js/auto_mode.js` + `css/auto_mode.css`** conectado a `js/game.js` vía `window.MimesisGameBridge` (reutilizando cámara, `captureSnapshotFromVideo`, `drawPoseOverlay`, `renderHeatmapTo`, `renderClassifierProbsTo` y `openInspectorModal`).
- **Fullscreen sobre `document.documentElement`** (con respaldo fijo 100% viewport) para que todos los modales compartidos (`#detail-report-modal`, `#inspector-modal`, `#pdi-lightbox`, `#toast`) se rendericen siempre por encima en pantalla completa.
- **Flujo automático por las 3 obras (2–4 jugadores):**
  1. **Preparación (`10s`):** nombre personalizado del jugador en barra superior delgada (sin tapar el cuerpo), escultura de referencia en esquina inferior derecha (o 3 miniaturas en Modo Libre), contador circular en esquina inferior izquierda, beeps Web Audio en `3-2-1`, sonido de obturador + flash de `180ms` al capturar.
  2. **Cambio de jugador (`5s`):** indica al jugador actual salir del encuadre y dar paso al siguiente mientras la foto recién tomada se analiza en segundo plano (`POST /api/analysis/pose` con `include_debug_views=true`).
  3. **Resultados de ronda (`8s`):** tarjetas comparativas por jugador (esqueleto, puntaje, feedback, mini mapa de calor) + ranking acumulado. Errores `HTTP 422` registran `0 pts` con aviso claro sin detener la partida.
  4. **Pausa e Informe Detallado (`#detail-report-modal`):** incluye botones **«⏸ Pausar / ▶ Continuar»**, **«Ver informe detallado»** (también por tarjeta) y **«Siguiente obra →»**. El modal trae pestañas por jugador, todas las métricas (`Segmentos / Mapa de calor / Escultura`, probabilidades ML, mapa de calor, 5 métricas PDI, contexto cultural y botón **«Ver informe PDI»**). Abrir el Informe Detallado (o el Inspector/Lightbox PDI) congela el reloj de `8s`; al cerrarlo, **reanuda desde el tiempo restante** si el usuario no había pausado manualmente.
  5. **Podio Final:** tabla con desglose por las 3 obras, ganador/empate y botones para revisar informes de cada ronda o volver a jugar.
- **Visor Lightbox del Pipeline PDI (`#pdi-lightbox`):** usable en **modo normal y pantalla completa**. Al hacer clic en cualquiera de las 5 tarjetas del Inspector PDI (`01 Original` a `05 Modo Escultura`, generadas a `max_side=960` px en `backend/pdi.py`), abre la imagen completa con contador `X / 5`, flechas `← / →` y atajos de teclado (`ArrowLeft`, `ArrowRight`, `Escape`).
- **Control de caché en desarrollo:** versionado `?v=20261124` en `game.html` + middleware `Cache-Control: no-cache, must-revalidate` en `backend/app.py` para `/`, `/index.html`, `/game.html`, `/js/*` y `/css/*`.

---

## 4. Resultados Cuantitativos Verificados (Para Informe y Póster)

### Ablación PDI (`notebooks/Mimesis_MLP_Entrenamiento.ipynb` · 329 fotos reales)
- **Sin PDI:** detección BlazePose `327 / 329` (`99.4%`), visibilidad media `0.942`.
- **Con PDI (CLAHE LAB + Letterboxing 640×640):** detección BlazePose **`328 / 329` (`99.7%`)**, visibilidad media **`0.942`** (`100%` en `val` `23/23` y `100%` en `test` `48/48`).

### Comparativa de Modelos (`Train: 258 [86×3] · Val: 23 [9/5/9] · Test: 48 [16×3]`)

| Modelo | Entrada | Val Accuracy (Macro-F1) | Test Accuracy (Macro-F1) |
| :--- | :--- | :---: | :---: |
| CNN RGB (Parcial 2 *baseline*) | Píxeles RGB | 83.3% (—) | 77.1% (—) |
| Random Forest | 12 ángulos + PDI | 91.3% (0.883) | 95.8% (0.959) |
| SVM (RBF) | 12 ángulos + PDI | 91.3% (0.883) | 97.9% (0.979) |
| **`PoseMLP` (PyTorch · producción)** | **12 ángulos + PDI** | **91.3% (0.883)** | **95.8% (0.959 · `46/48`)** |

- Desglose `PoseMLP` en `Test`: Belalcázar `16/16` (`F1 0.94`), Discóbolo `15/16` (`F1 0.97`), Shakespeare `15/16` (`F1 0.97`). Mejora de **`+8.0 pp` en `val`** y **`+18.7 pp` en `test`** frente a la CNN RGB.

### Pruebas Automatizadas (`backend/tests/test_pipeline.py`)
- **8/8 pruebas aprobadas (`OK`):** Laplaciano/CLAHE, letterbox/unletterbox (puntos y máscara), morfología + IoU/Hu en espejo, vistas debug PDI, `_compute_head_tilt` (frontal vs. perfil), invarianza espejo + Modo Libre, `test_body_framing_and_ood_non_pose_rejection` (plano cerrado sin rodillas, pose neutra, sentada y T-Pose) y prueba E2E real sobre `Fotos Prueba/Discobolo/20261015_132054.jpg` (`score=86.1`, `classify_pred=discobolo 100.0%`, `IoU=44.9%`, `Hu=40.6%`).

---

## 5. Mapa Actual de Archivos Clave

- **Frontend:** `index.html`, `game.html`, `css/{styles,layout,components,auto_mode}.css`, `js/{sculptures,lobby,game,auto_mode}.js`, `assets/sculptures/{belalcazar,discobolo,shakespeare}.jpg`.
- **Backend:** `backend/{app,pdi,pose,segmentation,evaluator,extract_references}.py`, `backend/requirements.txt`, `backend/tests/test_pipeline.py`.
- **Modelos y referencias:** `backend/models/{pose_mlp.pt,pose_landmarker_full.task}`, `backend/data/reference_poses.json`, `backend/data/reference_masks/*.png`, `backend/data/reference_previews/*_{skeleton,mask}.jpg`.
- **Notebooks:** `notebooks/Mimesis_MLP_Entrenamiento.ipynb` (entrenamiento oficial con soporte EXIF + `.heic`/`.jfif`, evaluación `val`+`test` y matrices de confusión), `Parcial2_Clasificacion_Poses_Esculturas.ipynb`, `Modelo_Segmentacion_Parcial3.ipynb`.
- **Documentación:** `README.md` (8 secciones de la rúbrica), `Desarrollo del sistema - Mimesis.pdf`, `Rubrica proyecto final PDI 2026-2.pdf`.

---

## 6. Entorno Local y Comandos de Ejecución

- **SO:** Windows · **Python verificado:** `3.13.14` (Microsoft Store; ejecutar módulos con `python -m ...`).
- **Importante (PowerShell):** siempre ubicarse primero en la carpeta del proyecto (con comillas por los espacios en la ruta) para evitar `ModuleNotFoundError: No module named 'backend'`:
  ```powershell
  cd "D:\Universidad\Procesamiento Digital de imagenes\ProyectoFinal\Mimesis"
  python -m uvicorn backend.app:app --reload
  ```
- **URL local:** `http://localhost:8000`
- **Pruebas automatizadas:**
  ```powershell
  python -m unittest backend.tests.test_pipeline -v
  ```
- **Regenerar poses y máscaras de referencia:**
  ```powershell
  python -m backend.extract_references
  ```
- **Nota de cámara móvil (Samsung/DroidCam):** si DroidCam da pantalla verde en Chrome, reiniciar DroidCam antes de abrir Chrome, probar Firefox o usar Iriun Webcam por USB.

---

## 7. Pendientes Reales Antes de la Entrega Final

1. **Enlace del dataset en `README.md` (línea 118):** reemplazar el placeholder por el enlace público de lectura de Google Drive (`PDI - Parcial2`).
2. **Outputs guardados en `notebooks/Mimesis_MLP_Entrenamiento.ipynb`:** actualmente tiene `0/14` celdas con salidas guardadas; subir la copia ejecutada en Colab para que las tablas, curvas y matrices de confusión sean visibles directamente en GitHub.
3. **Evidencias visuales y actualización de `README.md`:**
   - Incrustar visualizaciones en Datos (esculturas/máscaras/esqueletos de referencia), GUI (capturas reales del modo manual, modo automático en pantalla completa, Informe Detallado e Inspector PDI + Lightbox) y Resultados (matrices de confusión y ejemplo real).
   - Documentar en `README.md` los cambios recientes (`js/auto_mode.js`, `css/auto_mode.css`, modo automático, Lightbox PDI, validación de encuadre/OOD y `8/8` pruebas) e incluir guía de uso paso a paso.
4. **Pulido menor antes del `git push` final:**
   - Corregir `src="assets/sculptures/belalcazar.svg"` → `.jpg` en `index.html` (elimina un `404` inicial) y el comentario desactualizado *"480 fotos"* en `backend/evaluator.py`.
   - Agregar al menos 1 foto de prueba de *William Shakespeare* en `Fotos Prueba/Shakespeare/`.
   - Hacer `git commit` y `git push` de los cambios recientes al remoto (`https://github.com/xxjul123/Mimesis.git`).
5. **Entregables académicos externos:**
   - **Video demostrativo** (máx. 3 minutos).
   - **Póster académico** (siguiendo la Figura 1 de la rúbrica).
   - **Presentación y sustentación oral** con demo en vivo.
