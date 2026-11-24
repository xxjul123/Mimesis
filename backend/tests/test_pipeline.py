"""
test_pipeline.py — Pruebas automatizadas del pipeline completo de MIMESIS:
PDI (OpenCV) -> Segmentación morfológica -> Pose (BlazePose / 12 ángulos) ->
Evaluador (Modo Reto + Modo Libre + Espejo) -> Endpoint FastAPI.

Ejecución:
    python -m unittest backend.tests.test_pipeline -v
    (o con pytest: python -m pytest backend/tests/test_pipeline.py -v)
"""

from __future__ import annotations

import asyncio
import io
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from backend.app import analyze_pose, health
from backend.evaluator import PoseEvaluator, _mirror_angles
from backend.pdi import (
    apply_adaptive_clahe,
    compute_sharpness,
    generate_pdi_debug_views,
    letterbox_resize,
    unletterbox_mask,
    unletterbox_point,
)
from backend.pose import (
    FEATURE_ORDER,
    _compute_head_tilt,
    extract_geometric_features,
    validate_body_framing,
)
from backend.segmentation import compare_silhouettes, extract_main_contour, refine_mask

ROOT = Path(__file__).resolve().parent.parent.parent


def _make_dummy_landmarks(vis_left_ear: float = 0.95, vis_right_ear: float = 0.95):
    lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95} for _ in range(33)]
    lms[0] = {"x": 0.50, "y": 0.18, "z": 0.0, "visibility": 0.99}
    lms[7] = {"x": 0.45, "y": 0.17, "z": 0.0, "visibility": vis_left_ear}
    lms[8] = {"x": 0.55, "y": 0.17, "z": 0.0, "visibility": vis_right_ear}
    lms[11] = {"x": 0.40, "y": 0.30, "z": 0.0, "visibility": 0.99}
    lms[12] = {"x": 0.60, "y": 0.30, "z": 0.0, "visibility": 0.99}
    lms[13] = {"x": 0.32, "y": 0.42, "z": 0.0, "visibility": 0.99}
    lms[14] = {"x": 0.68, "y": 0.42, "z": 0.0, "visibility": 0.99}
    lms[15] = {"x": 0.28, "y": 0.55, "z": 0.0, "visibility": 0.99}
    lms[16] = {"x": 0.72, "y": 0.55, "z": 0.0, "visibility": 0.99}
    lms[23] = {"x": 0.44, "y": 0.58, "z": 0.0, "visibility": 0.99}
    lms[24] = {"x": 0.56, "y": 0.58, "z": 0.0, "visibility": 0.99}
    lms[25] = {"x": 0.43, "y": 0.76, "z": 0.0, "visibility": 0.99}
    lms[26] = {"x": 0.57, "y": 0.76, "z": 0.0, "visibility": 0.99}
    lms[27] = {"x": 0.43, "y": 0.93, "z": 0.0, "visibility": 0.99}
    lms[28] = {"x": 0.57, "y": 0.93, "z": 0.0, "visibility": 0.99}
    return lms


class TestMimesisPipeline(unittest.TestCase):
    def test_pdi_sharpness_and_clahe(self):
        flat = np.full((240, 320, 3), 50, dtype=np.uint8)
        lap_flat, is_blurry_flat = compute_sharpness(flat)
        self.assertTrue(is_blurry_flat)
        self.assertLess(lap_flat, 1.0)

        sharp = flat.copy()
        cv2.rectangle(sharp, (40, 40), (280, 200), (255, 255, 255), 3)
        cv2.line(sharp, (0, 0), (320, 240), (220, 220, 220), 2)
        lap_sharp, is_blurry_sharp = compute_sharpness(sharp)
        self.assertFalse(is_blurry_sharp)
        self.assertGreater(lap_sharp, 40.0)

        enhanced, mean_l, applied = apply_adaptive_clahe(flat)
        self.assertTrue(applied)
        self.assertLess(mean_l, 115.0)
        self.assertEqual(enhanced.shape, flat.shape)

    def test_letterbox_and_unletterbox_roundtrip(self):
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        boxed, meta = letterbox_resize(img, target_size=640)
        self.assertEqual(boxed.shape, (640, 640, 3))

        ox, oy = unletterbox_point(0.5, 0.5, meta)
        self.assertAlmostEqual(ox, 0.5, places=3)
        self.assertAlmostEqual(oy, 0.5, places=3)

        mask_box = np.zeros((640, 640), dtype=np.float32)
        mask_box[200:440, 100:540] = 1.0
        mask_orig = unletterbox_mask(mask_box, meta)
        self.assertEqual(mask_orig.shape, (400, 800))

    def test_segmentation_morphology_iou_and_hu(self):
        raw = np.zeros((240, 240), dtype=np.float32)
        raw[40:200, 95:135] = 0.9
        raw[70:95, 135:195] = 0.9
        raw[10:15, 10:15] = 0.95  # ruido aislado

        clean = refine_mask(raw, threshold=0.45)
        self.assertEqual(clean.dtype, np.uint8)
        self.assertEqual(int(clean[12, 12]), 0)
        self.assertIsNotNone(extract_main_contour(clean))

        same = compare_silhouettes(clean, clean)
        self.assertIsNotNone(same)
        self.assertGreaterEqual(same["iou"], 98.0)
        self.assertGreaterEqual(same["hu_similarity"], 95.0)

        mirrored = cv2.flip(clean, 1)
        mirror_cmp = compare_silhouettes(mirrored, clean)
        self.assertIsNotNone(mirror_cmp)
        self.assertTrue(mirror_cmp["mirrored"])
        self.assertGreaterEqual(mirror_cmp["iou"], 95.0)

    def test_pdi_debug_views_generation(self):
        img = np.full((300, 400, 3), 80, dtype=np.uint8)
        cv2.circle(img, (200, 150), 60, (200, 180, 160), -1)
        mask = np.zeros((300, 400), dtype=np.uint8)
        cv2.circle(mask, (200, 150), 60, 255, -1)

        views = generate_pdi_debug_views(img, img, mask=mask)
        for key in ("clahe_view", "laplacian_view", "mask_contour_view", "sculpture_mode_view"):
            self.assertIsInstance(views[key], str)
            self.assertTrue(views[key].startswith("data:image/jpeg;base64,"))

    def test_head_tilt_frontal_vs_profile_and_features(self):
        lms_front = _make_dummy_landmarks(vis_left_ear=0.95, vis_right_ear=0.95)
        pts_front = np.array([[lm["x"], lm["y"]] for lm in lms_front], dtype=np.float32)
        self.assertLess(_compute_head_tilt(lms_front, pts_front), 1.0)

        lms_prof = _make_dummy_landmarks(vis_left_ear=0.0, vis_right_ear=0.95)
        lms_prof[8]["y"] = 0.14
        lms_prof[0]["y"] = 0.22
        pts_prof = np.array([[lm["x"], lm["y"]] for lm in lms_prof], dtype=np.float32)
        self.assertGreater(_compute_head_tilt(lms_prof, pts_prof), 30.0)

        angles, vec = extract_geometric_features(lms_front)
        self.assertEqual(len(angles), len(FEATURE_ORDER))
        self.assertEqual(vec.shape, (12,))
        self.assertGreaterEqual(float(vec.min()), 0.0)
        self.assertLessEqual(float(vec.max()), 1.0)

    def test_evaluator_mirror_invariance_and_classify_mode(self):
        ev = PoseEvaluator()
        for sid, ref_angles in ev.references.items():
            res_normal = ev.evaluate(ref_angles, sculpture_id=sid, game_mode="challenge")
            res_mirror = ev.evaluate(_mirror_angles(ref_angles), sculpture_id=sid, game_mode="challenge")
            self.assertAlmostEqual(res_normal["score"], res_mirror["score"], places=5)
            self.assertAlmostEqual(sum(res_normal["class_probabilities"].values()), 100.0, delta=0.5)

            res_free = ev.evaluate(ref_angles, game_mode="classify")
            self.assertEqual(res_free["predicted_class"], sid)
            self.assertEqual(res_free["sculpture_id"], sid)
            self.assertTrue(res_free["is_recognized"])

    def test_body_framing_and_ood_non_pose_rejection(self):
        # 1. Encuadre completo válido (rodillas visibles) vs plano cerrado sin piernas
        full_lms = _make_dummy_landmarks()
        ok_full, _ = validate_body_framing(full_lms)
        self.assertTrue(ok_full)

        closeup_lms = _make_dummy_landmarks()
        for idx in (25, 26, 27, 28):
            closeup_lms[idx]["visibility"] = 0.05
            closeup_lms[idx]["y"] = 1.35
        ok_close, msg_close = validate_body_framing(closeup_lms)
        self.assertFalse(ok_close)
        self.assertIn("rodillas", msg_close.lower())

        # 2. Postura neutra parado (brazos caídos sin imitar obra) rechazada en Modo Libre
        ev = PoseEvaluator()
        neutral_angles = {
            "head_tilt": 2.0, "neck_angle": 4.0,
            "torso_lean": 2.0, "shoulder_level": 2.0,
            "left_elbow": 170.0, "right_elbow": 170.0,
            "left_shoulder": 10.0, "right_shoulder": 10.0,
            "left_hip": 176.0, "right_hip": 176.0,
            "left_knee": 176.0, "right_knee": 176.0,
        }
        res_neutral = ev.evaluate(neutral_angles, game_mode="classify")
        self.assertFalse(res_neutral["is_recognized"])
        self.assertEqual(res_neutral["score"], 0.0)

        # 3. Postura sentada / fuera de distribución (geo bajo) rechazada en Modo Libre y sin inflar en Modo Reto
        seated_angles = {
            "head_tilt": 15.0, "neck_angle": 18.0,
            "torso_lean": 12.0, "shoulder_level": 8.0,
            "left_elbow": 110.0, "right_elbow": 135.0,
            "left_shoulder": 28.0, "right_shoulder": 35.0,
            "left_hip": 95.0, "right_hip": 105.0,
            "left_knee": 95.0, "right_knee": 100.0,
        }
        res_seated_free = ev.evaluate(seated_angles, game_mode="classify")
        self.assertFalse(res_seated_free["is_recognized"])
        self.assertEqual(res_seated_free["score"], 0.0)

        res_seated_challenge = ev.evaluate(seated_angles, sculpture_id="discobolo", game_mode="challenge")
        self.assertLess(res_seated_challenge["score"], 45.0)

        # 4. T-Pose (ambos brazos en cruz) rechazada en Modo Libre y sin inflar en Modo Reto
        tpose_angles = {
            "head_tilt": 15.0, "neck_angle": 18.0,
            "torso_lean": 3.0, "shoulder_level": 4.0,
            "left_elbow": 175.0, "right_elbow": 175.0,
            "left_shoulder": 92.0, "right_shoulder": 92.0,
            "left_hip": 174.0, "right_hip": 174.0,
            "left_knee": 175.0, "right_knee": 175.0,
        }
        res_tpose_free = ev.evaluate(tpose_angles, game_mode="classify")
        self.assertFalse(res_tpose_free["is_recognized"])
        self.assertEqual(res_tpose_free["score"], 0.0)
        res_tpose_challenge = ev.evaluate(tpose_angles, sculpture_id="belalcazar", game_mode="challenge")
        self.assertLess(res_tpose_challenge["score"], 35.0)

    def test_fastapi_health_and_real_photo_end_to_end(self):
        body = health()
        self.assertEqual(body["status"], "ok")
        self.assertTrue({"belalcazar", "discobolo", "shakespeare"}.issubset(set(body["sculptures"])))
        self.assertTrue({"belalcazar", "discobolo", "shakespeare"}.issubset(set(body["reference_masks_loaded"])))

        sample_photo = ROOT / "Fotos Prueba" / "Discobolo" / "20261015_132054.jpg"
        if sample_photo.exists():
            raw = sample_photo.read_bytes()

            up1 = UploadFile(filename="test.jpg", file=io.BytesIO(raw))
            d1 = asyncio.run(
                analyze_pose(
                    image=up1,
                    player="Tester",
                    sculpture_id="discobolo",
                    sculpture=None,
                    round_num="1",
                    game_mode="challenge",
                    include_debug_views=True,
                )
            )
            self.assertGreaterEqual(d1["score"], 0.0)
            self.assertLessEqual(d1["score"], 100.0)
            self.assertIsNotNone(d1["silhouette_metrics"])
            self.assertIn("iou", d1["silhouette_metrics"])
            self.assertIn("hu_similarity", d1["silhouette_metrics"])
            self.assertIsNotNone(d1["debug_views"])
            self.assertTrue(d1["debug_views"]["sculpture_mode_view"].startswith("data:image/jpeg;base64,"))

            up2 = UploadFile(filename="test.jpg", file=io.BytesIO(raw))
            d2 = asyncio.run(
                analyze_pose(
                    image=up2,
                    player="Tester",
                    sculpture_id=None,
                    sculpture=None,
                    round_num="1",
                    game_mode="classify",
                    include_debug_views=False,
                )
            )
            self.assertEqual(d2["game_mode"], "classify")
            self.assertIn(d2["predicted_class"], {"belalcazar", "discobolo", "shakespeare"})
            self.assertEqual(set(d2["class_probabilities"].keys()), {"belalcazar", "discobolo", "shakespeare"})
            print(
                f"\n[E2E Discobolo] challenge_score={d1['score']} | "
                f"IoU={d1['silhouette_metrics']['iou']}% | Hu={d1['silhouette_metrics']['hu_similarity']}% | "
                f"classify_pred={d2['predicted_class']} ({d2['predicted_confidence']}%)"
            )


if __name__ == "__main__":
    unittest.main()
