"""
extract_references.py — Extrae los 33 landmarks y los 12 ángulos de referencia
directamente desde las 3 fotografías reales de las esculturas en assets/sculptures/.

Nota técnica:
- MediaPipe BlazePose fue entrenado sobre humanos reales. En estatuas de mármol
  o bronce oscuro, algunas extremidades pueden desviarse (ej. el brazo levantado del
  Discóbolo o la pierna doblada de Belalcázar sobre el bronce oscuro).
- Por eso aplicamos la extracción base con BlazePose y un refinamiento supervisado
  sobre esos puntos puntuales, verificando visualmente el esqueleto resultante en
  backend/data/reference_previews/.

Uso desde la raíz del proyecto:
    python -m backend.extract_references
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from .pdi import decode_image, letterbox_resize
from .pose import PoseExtractor, extract_geometric_features

ROOT = Path(__file__).resolve().parent.parent
SCULPTURES_DIR = ROOT / "assets" / "sculptures"
OUT_JSON = Path(__file__).parent / "data" / "reference_poses.json"
PREVIEW_DIR = Path(__file__).parent / "data" / "reference_previews"

# Colores BGR por segmento corporal para las previsualizaciones
SEGMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cabeza": (30, 215, 255),      # Amarillo dorado
    "torso": (235, 235, 235),      # Blanco / plata
    "brazo_izq": (255, 215, 50),   # Cian
    "brazo_der": (65, 110, 255),   # Naranja / coral
    "pierna_izq": (95, 235, 115),  # Verde brillante
    "pierna_der": (235, 100, 220), # Magenta / violeta
}

SEGMENT_LEGEND_NAMES: Dict[str, str] = {
    "cabeza": "Cabeza",
    "torso": "Torso",
    "brazo_izq": "Brazo Izq",
    "brazo_der": "Brazo Der",
    "pierna_izq": "Pierna Izq",
    "pierna_der": "Pierna Der",
}

# Conexiones agrupadas por segmento corporal (solo puntos usados para calificar)
SEGMENT_CONNECTIONS: List[Tuple[int, int, str]] = [
    # Cabeza (nariz + orejas conectadas a hombros)
    (0, 7, "cabeza"),
    (0, 8, "cabeza"),
    (7, 11, "cabeza"),
    (8, 12, "cabeza"),
    # Torso
    (11, 12, "torso"),
    (11, 23, "torso"),
    (12, 24, "torso"),
    (23, 24, "torso"),
    # Brazo izquierdo
    (11, 13, "brazo_izq"),
    (13, 15, "brazo_izq"),
    # Brazo derecho
    (12, 14, "brazo_der"),
    (14, 16, "brazo_der"),
    # Pierna izquierda
    (23, 25, "pierna_izq"),
    (25, 27, "pierna_izq"),
    # Pierna derecha
    (24, 26, "pierna_der"),
    (26, 28, "pierna_der"),
]

# Los 15 landmarks de BlazePose usados realmente para calcular los 12 ángulos en pose.py.
# Los 33 landmarks completos se siguen guardando en reference_poses.json, pero en las
# previsualizaciones solo se muestran estos 15 puntos clave con su etiqueta y color.
SCORING_LANDMARKS: Dict[int, Tuple[str, str]] = {
    0: ("0:Nariz", "cabeza"),
    7: ("7:Oreja I", "cabeza"),
    8: ("8:Oreja D", "cabeza"),
    11: ("11:Hombro I", "brazo_izq"),
    12: ("12:Hombro D", "brazo_der"),
    13: ("13:Codo I", "brazo_izq"),
    14: ("14:Codo D", "brazo_der"),
    15: ("15:Muneca I", "brazo_izq"),
    16: ("16:Muneca D", "brazo_der"),
    23: ("23:Cadera I", "pierna_izq"),
    24: ("24:Cadera D", "pierna_der"),
    25: ("25:Rodilla I", "pierna_izq"),
    26: ("26:Rodilla D", "pierna_der"),
    27: ("27:Tobillo I", "pierna_izq"),
    28: ("28:Tobillo D", "pierna_der"),
}

# Correcciones supervisadas (en coordenadas normalizadas [0, 1]) para extremidades
# donde BlazePose se desvía por tratarse de bronce/mármol en lugar de piel humana.
SUPERVISED_OVERRIDES: Dict[str, Dict[int, Tuple[float, float]]] = {
    "discobolo":{
        14: (0.410, 0.115),  # codo derecho levantado hacia el disco
        16: (0.255, 0.055),  # muñeca derecha sosteniendo el disco
        18: (0.230, 0.045),
        20: (0.225, 0.040),
        22: (0.240, 0.050),
        13: (0.695, 0.465),  # codo izquierdo
        15: (0.585, 0.615),  # muñeca izquierda (sobre la rodilla)
    },
    "belalcazar": {
        12: (0.432, 0.272),  # hombro derecho (brazo que señala)
        14: (0.340, 0.278),  # codo derecho
        16: (0.222, 0.255),  # muñeca derecha
        18: (0.175, 0.245),
        20: (0.165, 0.242),
        22: (0.185, 0.250),
        11: (0.538, 0.285),  # hombro izquierdo (brazo sobre la espada)
        13: (0.562, 0.345),  # codo izquierdo
        15: (0.578, 0.378),  # muñeca izquierda en la empuñadura
        17: (0.582, 0.395),
        19: (0.580, 0.395),
        21: (0.575, 0.390),
        24: (0.468, 0.485),  # cadera derecha (pierna recta)
        26: (0.456, 0.618),  # rodilla derecha
        28: (0.445, 0.735),  # tobillo derecho
        30: (0.432, 0.752),
        32: (0.485, 0.762),
        23: (0.535, 0.485),  # cadera izquierda (pierna doblada sobre roca)
        25: (0.625, 0.555),  # rodilla izquierda flexionada
        27: (0.518, 0.615),  # tobillo izquierdo apoyado en bloque
        29: (0.502, 0.612),
        31: (0.542, 0.665),
    },
}


def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int = 4) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _draw_legend(vis_img, scale: float) -> Tuple[int, int, int, int]:
    """Dibuja una leyenda compacta en la esquina superior izquierda y retorna su bbox."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, 0.52 * scale)
    thickness = max(1, int(round(1.5 * scale)))
    pad_x = max(10, int(round(12 * scale)))
    pad_y = max(8, int(round(10 * scale)))
    swatch_r = max(5, int(round(6 * scale)))
    row_gap = max(6, int(round(8 * scale)))

    entries = [(SEGMENT_LEGEND_NAMES[seg], color) for seg, color in SEGMENT_COLORS.items()]
    text_sizes = [cv2.getTextSize(name, font, font_scale, thickness)[0] for name, _ in entries]
    max_tw = max(tw for tw, _ in text_sizes)
    row_h = max(th for _, th in text_sizes) + row_gap

    x0 = pad_x
    y0 = pad_y
    box_w = pad_x * 2 + swatch_r * 2 + int(round(10 * scale)) + max_tw
    box_h = pad_y * 2 + row_h * len(entries) - row_gap // 2
    x1, y1 = x0 + box_w, y0 + box_h

    overlay = vis_img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 17, 15), -1)
    cv2.addWeighted(overlay, 0.78, vis_img, 0.22, 0, vis_img)
    cv2.rectangle(vis_img, (x0, y0), (x1, y1), (110, 110, 110), max(1, int(round(1.2 * scale))), cv2.LINE_AA)

    cur_y = y0 + pad_y
    for (name, color), (_, th) in zip(entries, text_sizes):
        cy = cur_y + th // 2
        cx = x0 + pad_x + swatch_r
        cv2.circle(vis_img, (cx, cy), swatch_r, color, -1, cv2.LINE_AA)
        cv2.circle(vis_img, (cx, cy), swatch_r, (18, 17, 15), max(1, int(round(1.2 * scale))), cv2.LINE_AA)
        tx = cx + swatch_r + int(round(10 * scale))
        ty = cur_y + th
        cv2.putText(vis_img, name, (tx, ty), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
        cur_y += row_h

    return (x0, y0, x1, y1)


def draw_preview(img_bgr, landmarks: List[Dict[str, float]], out_path: Path):
    """
    Genera la imagen de previsualización mostrando ÚNICAMENTE los 15 puntos clave
    usados para calificar, diferenciados por color de segmento y con etiquetas legibles.
    """
    vis_img = img_bgr.copy()
    h, w = vis_img.shape[:2]
    scale = max(0.65, max(h, w) / 1400.0)

    line_w = max(3, int(round(4.5 * scale)))
    outline_w = line_w + max(2, int(round(3.0 * scale)))
    pt_r = max(5, int(round(7.0 * scale)))
    pt_border = max(2, int(round(2.0 * scale)))

    # 1. Dibujar conexiones coloreadas por segmento (con borde oscuro para contraste)
    for a, b, segment in SEGMENT_CONNECTIONS:
        pa, pb = landmarks[a], landmarks[b]
        if pa["visibility"] < 0.3 or pb["visibility"] < 0.3:
            continue
        p1 = (int(pa["x"] * w), int(pa["y"] * h))
        p2 = (int(pb["x"] * w), int(pb["y"] * h))
        color = SEGMENT_COLORS[segment]
        cv2.line(vis_img, p1, p2, (18, 17, 15), outline_w, cv2.LINE_AA)
        cv2.line(vis_img, p1, p2, color, line_w, cv2.LINE_AA)

    # 2. Dibujar únicamente los 15 puntos de calificación
    visible_points: List[Tuple[int, str, str, int, int]] = []
    for idx, (label, segment) in SCORING_LANDMARKS.items():
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if lm["visibility"] < 0.3:
            continue
        px, py = int(lm["x"] * w), int(lm["y"] * h)
        color = SEGMENT_COLORS[segment]
        cv2.circle(vis_img, (px, py), pt_r, color, -1, cv2.LINE_AA)
        cv2.circle(vis_img, (px, py), pt_r, (18, 17, 15), pt_border, cv2.LINE_AA)
        visible_points.append((idx, label, segment, px, py))

    # 3. Leyenda en esquina superior izquierda
    legend_box = _draw_legend(vis_img, scale)

    # 4. Etiquetas legibles sin solapamiento para cada punto visible
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, 0.48 * scale)
    text_thick = max(1, int(round(1.4 * scale)))
    pad_x = max(5, int(round(6 * scale)))
    pad_y = max(3, int(round(4 * scale)))
    margin = max(6, int(round(8 * scale)))

    # Centro horizontal aproximado del cuerpo para orientar etiquetas hacia afuera
    body_cx = sum(px for _, _, _, px, _ in visible_points) / max(1, len(visible_points))

    placed_boxes: List[Tuple[int, int, int, int]] = [legend_box]
    # Reservar un área pequeña alrededor de cada punto para no tapar los círculos
    point_boxes = [(px - pt_r - 2, py - pt_r - 2, px + pt_r + 2, py + pt_r + 2) for _, _, _, px, py in visible_points]

    # Procesar primero cabeza (suelen estar muy juntos) y luego resto de arriba a abajo
    ordered_points = sorted(visible_points, key=lambda item: (0 if item[2] == "cabeza" else 1, item[4], item[3]))

    for idx, label, segment, px, py in ordered_points:
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thick)
        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2 + baseline // 2
        color = SEGMENT_COLORS[segment]

        # Candidatos de desplazamiento (dx, dy) priorizando lado exterior del cuerpo
        prefer_right = px >= body_cx
        horiz_dirs = [1, -1] if prefer_right else [-1, 1]
        base_d = max(14, int(round(18 * scale)))

        candidates: List[Tuple[int, int]] = []
        for mult in (1.0, 2.2, 3.6, 5.2, 7.0):
            d = int(round(base_d * mult))
            for hdir in horiz_dirs:
                bx = px + d if hdir > 0 else px - d - box_w
                candidates.append((bx, py - box_h // 2))
                candidates.append((bx, py - box_h - int(d * 0.45)))
                candidates.append((bx, py + int(d * 0.45)))
            candidates.append((px - box_w // 2, py - d - box_h))
            candidates.append((px - box_w // 2, py + d))

        chosen_box: Tuple[int, int, int, int] | None = None
        for bx, by in candidates:
            x0 = max(margin, min(w - margin - box_w, bx))
            y0 = max(margin, min(h - margin - box_h, by))
            cand = (x0, y0, x0 + box_w, y0 + box_h)
            if any(_boxes_overlap(cand, pb, pad=4) for pb in placed_boxes):
                continue
            if any(_boxes_overlap(cand, ptb, pad=2) for ptb in point_boxes):
                continue
            chosen_box = cand
            break

        if chosen_box is None:
            # Respaldo: colocar evitando al menos otras etiquetas
            for bx, by in candidates:
                x0 = max(margin, min(w - margin - box_w, bx))
                y0 = max(margin, min(h - margin - box_h, by))
                cand = (x0, y0, x0 + box_w, y0 + box_h)
                if not any(_boxes_overlap(cand, pb, pad=2) for pb in placed_boxes):
                    chosen_box = cand
                    break

        if chosen_box is None:
            x0 = max(margin, min(w - margin - box_w, px + base_d))
            y0 = max(margin, min(h - margin - box_h, py - box_h // 2))
            chosen_box = (x0, y0, x0 + box_w, y0 + box_h)

        placed_boxes.append(chosen_box)
        x0, y0, x1, y1 = chosen_box

        # Línea guía desde el punto hasta el borde más cercano de la etiqueta
        anchor_x = max(x0, min(x1, px))
        anchor_y = max(y0, min(y1, py))
        cv2.line(vis_img, (px, py), (anchor_x, anchor_y), (18, 17, 15), max(2, int(round(2.5 * scale))), cv2.LINE_AA)
        cv2.line(vis_img, (px, py), (anchor_x, anchor_y), color, max(1, int(round(1.3 * scale))), cv2.LINE_AA)

        # Fondo semitransparente oscuro + borde con el color del segmento
        roi = vis_img[y0:y1, x0:x1]
        dark = roi.copy()
        dark[:] = (18, 17, 15)
        cv2.addWeighted(dark, 0.80, roi, 0.20, 0, roi)
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, max(1, int(round(1.5 * scale))), cv2.LINE_AA)

        text_org = (x0 + pad_x, y0 + pad_y + th)
        cv2.putText(vis_img, label, text_org, font, font_scale, (248, 248, 248), text_thick, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_img)


def extract_for_sculpture(extractor: PoseExtractor, sid: str, img_bgr):
    h, w = img_bgr.shape[:2]
    if sid == "belalcazar":
        # Recorte ROI alrededor de la estatua para separar del fondo de árboles
        crop = img_bgr[int(h * 0.12) : int(h * 0.82), int(w * 0.10) : int(w * 0.70)]
        boxed, meta = letterbox_resize(crop, 640)
        res = extractor.extract(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), meta)
        if res is None:
            return None
        lms, vis = res
        full_lms = []
        for lm in lms:
            full_lms.append(
                {
                    "x": round(0.10 + lm["x"] * 0.60, 5),
                    "y": round(0.12 + lm["y"] * 0.70, 5),
                    "z": lm["z"],
                    "visibility": lm["visibility"],
                }
            )
        lms = full_lms
    else:
        boxed, meta = letterbox_resize(img_bgr, 640)
        res = extractor.extract(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), meta)
        if res is None:
            return None
        lms, vis = res

    # Aplicar correcciones supervisadas si existen para esta estatua
    for idx, (nx, ny) in SUPERVISED_OVERRIDES.get(sid, {}).items():
        lms[idx]["x"] = round(nx, 5)
        lms[idx]["y"] = round(ny, 5)
        lms[idx]["visibility"] = 0.99

    return lms, vis


def main():
    extractor = PoseExtractor(model_complexity=1, min_detection_confidence=0.05)
    output = {
        "_comment": "Landmarks y ángulos reales extraídos de las 3 fotografías de esculturas (BlazePose + refinamiento supervisado en bronce/mármol)."
    }

    for sid in ("belalcazar", "discobolo", "shakespeare"):
        img_path = SCULPTURES_DIR / f"{sid}.jpg"
        if not img_path.exists():
            print(f"[ERROR] No existe {img_path}")
            continue

        img_bgr = decode_image(img_path.read_bytes())
        res = extract_for_sculpture(extractor, sid, img_bgr)
        if res is None:
            print(f"[AVISO] No se pudo extraer pose en {sid}.jpg")
            continue

        landmarks, mean_vis = res
        angles, _ = extract_geometric_features(landmarks)
        output[sid] = {
            "angles": {k: round(v, 2) for k, v in angles.items()},
            "landmarks": landmarks,
            "mean_visibility": round(mean_vis, 3),
        }
        draw_preview(img_bgr, landmarks, PREVIEW_DIR / f"{sid}_skeleton.jpg")
        print(f"[OK] {sid}: ángulos = {output[sid]['angles']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {OUT_JSON}")


if __name__ == "__main__":
    main()
