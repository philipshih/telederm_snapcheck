from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class AugmentationResult:
    image: Image.Image
    labels: Dict[str, int]
    metadata: Dict[str, float]


class QualityAugmentor:
    """Apply dermatology-specific quality degradations with bookkeeping."""

    def __init__(self, config: Dict[str, any], seed: int | None = None) -> None:
        self.config = config
        self.rng = random.Random(seed)

    def _to_cv(self, image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def _gaussian_blur(self, image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def _motion_blur(self, image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        slope = math.tan(math.radians(angle))
        for i in range(kernel_size):
            offset = int(round((i - center) * slope))
            x = center + offset
            y = i
            if 0 <= x < kernel_size:
                kernel[y, x] = 1
        kernel /= kernel.sum() if kernel.sum() > 0 else 1
        return cv2.filter2D(image, -1, kernel)

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.normal(0, std * 255, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _add_shadow(self, image: np.ndarray, intensity: float, softness: float, tint: Tuple[float, float, float]) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        num_blobs = self.rng.randint(1, 2)
        for _ in range(num_blobs):
            center_x = self.rng.randint(0, w)
            center_y = self.rng.randint(0, h)
            axis_x = max(1, int(self.rng.uniform(0.2, 0.6) * w))
            axis_y = max(1, int(self.rng.uniform(0.2, 0.6) * h))
            angle = self.rng.uniform(0, 180)
            ellipse = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                ellipse,
                (center_x, center_y),
                (axis_x, axis_y),
                angle,
                0,
                360,
                255,
                -1,
            )
            blur_size = int(max(h, w) * softness)
            if blur_size % 2 == 0:
                blur_size += 1
            blurred = cv2.GaussianBlur(ellipse.astype(np.float32), (blur_size, blur_size), 0)
            mask = np.maximum(mask, blurred / 255.0)
        shadow = np.clip(mask * intensity, 0, 1)[..., None]
        tinted = (1 - shadow) + shadow * np.array(tint, dtype=np.float32).reshape(1, 1, 3)
        return np.clip(image.astype(np.float32) * tinted, 0, 255).astype(np.uint8)

    def _add_obstruction(self, image: np.ndarray, size: Tuple[float, float]) -> np.ndarray:
        h, w = image.shape[:2]
        oh = int(h * size[0])
        ow = int(w * size[1])
        top = self.rng.randint(0, h - oh) if h - oh > 0 else 0
        left = self.rng.randint(0, w - ow) if w - ow > 0 else 0
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (left + ow // 2, top + oh // 2)
        axes = (max(1, ow // 2), max(1, oh // 2))
        angle = self.rng.uniform(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        blur_size = max(3, int(max(axes) * 0.8))
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0) / 255.0
        palettes = [
            (self.rng.randint(170, 220), self.rng.randint(140, 190), self.rng.randint(120, 170)),  # finger
            (self.rng.randint(30, 80), self.rng.randint(30, 80), self.rng.randint(30, 90)),        # phone border
            (self.rng.randint(200, 255), self.rng.randint(200, 255), self.rng.randint(200, 255)),  # gauze/light
        ]
        color = np.array(palettes[self.rng.randint(0, len(palettes) - 1)], dtype=np.float32)
        base = image.astype(np.float32)
        overlay = np.tile(color, (h, w, 1))
        return np.clip(base * (1 - mask[..., None]) + overlay * mask[..., None], 0, 255).astype(np.uint8)

    def _reframe(self, image: np.ndarray, crop_ratio: float) -> np.ndarray:
        h, w = image.shape[:2]
        ch = int(h * crop_ratio)
        cw = int(w * crop_ratio)
        top = (h - ch) // 2
        left = (w - cw) // 2
        cropped = image[top : top + ch, left : left + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _downscale(self, image: np.ndarray, factor: float) -> np.ndarray:
        h, w = image.shape[:2]
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        low_res = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_NEAREST)

    def _zoom_out(self, image: np.ndarray, scale: float, background: str = "mean") -> np.ndarray:
        h, w = image.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(image)
        if background == "mean":
            mean_color = image.reshape(-1, 3).mean(axis=0)
            canvas[..., :] = mean_color
        else:
            canvas[..., :] = np.array([200, 200, 200], dtype=np.uint8)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _zoom_in(self, image: np.ndarray, crop_scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        ch = max(10, int(h * crop_scale))
        cw = max(10, int(w * crop_scale))
        max_top = max(1, h - ch)
        max_left = max(1, w - cw)
        top = self.rng.randint(0, max_top)
        left = self.rng.randint(0, max_left)
        cropped = image[top : top + ch, left : left + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _color_cast(self, image: np.ndarray, shift: float) -> np.ndarray:
        factors = np.array(
            [
                self.rng.uniform(1 - shift, 1 + shift),
                self.rng.uniform(1 - shift, 1 + shift),
                self.rng.uniform(1 - shift, 1 + shift),
            ],
            dtype=np.float32,
        )
        cast = image.astype(np.float32) * factors.reshape(1, 1, 3)
        return np.clip(cast, 0, 255).astype(np.uint8)

    def _jpeg_compress(self, image: np.ndarray, quality: int) -> np.ndarray:
        ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return image
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else image

    def _apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        h, w = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w * strength)
        kernel_y = cv2.getGaussianKernel(h, h * strength)
        mask = kernel_y * kernel_x.T
        mask = mask / mask.max()
        vignette = image.astype(np.float32) * mask[..., None]
        return np.clip(vignette, 0, 255).astype(np.uint8)

    def _add_glare(self, image: np.ndarray, radius: float, intensity: float) -> np.ndarray:
        h, w = image.shape[:2]
        cx = self.rng.randint(int(w * 0.1), int(w * 0.9))
        cy = self.rng.randint(int(h * 0.1), int(h * 0.9))
        r = max(1, int(min(h, w) * radius))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), r * 0.5)
        highlight = np.clip(image.astype(np.float32) + intensity * 255 * mask[..., None], 0, 255)
        return highlight.astype(np.uint8)

    def _add_hair(self, image: np.ndarray, count: int, thickness_range: Tuple[int, int]) -> np.ndarray:
        overlay = image.copy()
        h, w = image.shape[:2]
        for _ in range(count):
            segments = self.rng.randint(3, 5)
            points = [
                (
                    self.rng.randint(0, w - 1),
                    self.rng.randint(0, h - 1),
                )
            ]
            for _ in range(segments):
                dx = self.rng.randint(-w // 6, w // 6)
                dy = self.rng.randint(-h // 6, h // 6)
                nx = int(np.clip(points[-1][0] + dx, 0, w - 1))
                ny = int(np.clip(points[-1][1] + dy, 0, h - 1))
                points.append((nx, ny))
            color = (
                self.rng.randint(15, 60),
                self.rng.randint(15, 60),
                self.rng.randint(15, 60),
            )
            thickness = self.rng.randint(thickness_range[0], thickness_range[1])
            cv2.polylines(
                overlay,
                [np.array(points, dtype=np.int32)],
                False,
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    def apply(self, pil_image: Image.Image) -> AugmentationResult:
        cv_image = self._to_cv(pil_image)
        labels: Dict[str, int] = {
            "blur": 0,
            "motion_blur": 0,
            "low_brightness": 0,
            "high_brightness": 0,
            "low_contrast": 0,
            "high_contrast": 0,
            "noise": 0,
            "shadow": 0,
            "obstruction": 0,
            "framing": 0,
            "low_resolution": 0,
            "zoom_out": 0,
            "zoom_in": 0,
            "color_cast": 0,
            "compression": 0,
            "vignette": 0,
            "glare": 0,
            "hair": 0,
            "overall_fail": 0,
        }
        metadata: Dict[str, float] = {}

        applied: List[str] = []
        defect_config = self.config.get("quality_defects", {})
        if not defect_config:
            return AugmentationResult(image=self._to_pil(cv_image), labels=labels, metadata=metadata)

        max_per_image = max(1, int(self.config.get("max_augmentations_per_image", 3)))
        min_per_image = int(self.config.get("min_augmentations_per_image", 1))
        if min_per_image > max_per_image:
            min_per_image = max_per_image
        count = self.rng.randint(min_per_image, max_per_image)

        defects = list(defect_config.keys())
        if not defects:
            return AugmentationResult(image=self._to_pil(cv_image), labels=labels, metadata=metadata)

        weights_cfg = self.config.get("defect_weights", {})
        if weights_cfg:
            # Weighted without replacement sampling
            selected: List[str] = []
            available = defects.copy()
            for _ in range(min(count, len(available))):
                total_weight = sum(float(weights_cfg.get(name, 1.0)) for name in available)
                pick = self.rng.random() * total_weight
                cumulative = 0.0
                chosen = available[0]
                for name in available:
                    cumulative += float(weights_cfg.get(name, 1.0))
                    if pick <= cumulative:
                        chosen = name
                        break
                selected.append(chosen)
                available.remove(chosen)
            defects = selected
        else:
            self.rng.shuffle(defects)
            defects = defects[:count]

        for defect in defects:
            if defect == "blur":
                kernel = self.rng.choice(defect_config[defect]["kernel_sizes"])
                sigma = self.rng.uniform(*defect_config[defect]["sigma_range"])
                cv_image = self._gaussian_blur(cv_image, kernel, sigma)
                labels["blur"] = 1
                metadata["blur_kernel"] = float(kernel)
                metadata["blur_sigma"] = float(sigma)
            elif defect == "motion_blur":
                kernel = self.rng.choice(defect_config[defect]["kernel_sizes"])
                angle = self.rng.uniform(*defect_config[defect]["angle_range"])
                cv_image = self._motion_blur(cv_image, kernel, angle)
                labels["motion_blur"] = 1
                metadata["motion_kernel"] = float(kernel)
                metadata["motion_angle"] = float(angle)
            elif defect == "brightness":
                factor = self.rng.uniform(*defect_config[defect]["factor_range"])
                cv_image = self._adjust_brightness(cv_image, factor)
                if factor < 1:
                    labels["low_brightness"] = 1
                else:
                    labels["high_brightness"] = 1
                metadata["brightness_factor"] = float(factor)
            elif defect == "contrast":
                factor = self.rng.uniform(*defect_config[defect]["factor_range"])
                cv_image = self._adjust_contrast(cv_image, factor)
                if factor < 1:
                    labels["low_contrast"] = 1
                else:
                    labels["high_contrast"] = 1
                metadata["contrast_factor"] = float(factor)
            elif defect == "noise":
                std = self.rng.uniform(*defect_config[defect]["std_range"])
                cv_image = self._add_noise(cv_image, std)
                labels["noise"] = 1
                metadata["noise_std"] = float(std)
            elif defect == "shadow":
                intensity = self.rng.uniform(*defect_config[defect]["intensity_range"])
                softness = self.rng.uniform(*defect_config[defect]["softness_range"])
                tint_low, tint_high = defect_config[defect]["tint_range"]
                tint = tuple(self.rng.uniform(tint_low, tint_high) for _ in range(3))
                cv_image = self._add_shadow(cv_image, intensity, softness, tint)
                labels["shadow"] = 1
                metadata["shadow_intensity"] = float(intensity)
                metadata["shadow_softness"] = float(softness)
                metadata["shadow_tint"] = float(np.mean(tint))
            elif defect == "obstruction":
                size = self.rng.choice(defect_config[defect]["patch_sizes"])
                cv_image = self._add_obstruction(cv_image, tuple(size))
                labels["obstruction"] = 1
                metadata["obstruction_height_ratio"] = float(size[0])
                metadata["obstruction_width_ratio"] = float(size[1])
            elif defect == "framing":
                crop_ratio = self.rng.uniform(*defect_config[defect]["crop_range"])
                cv_image = self._reframe(cv_image, crop_ratio)
                labels["framing"] = 1
                metadata["framing_ratio"] = float(crop_ratio)
            elif defect == "resolution":
                factor = self.rng.choice(defect_config[defect]["downscale_factors"])
                cv_image = self._downscale(cv_image, factor)
                labels["low_resolution"] = 1
                metadata["downscale_factor"] = float(factor)
            elif defect == "zoom_out":
                scale = self.rng.uniform(*defect_config[defect]["scale_range"])
                cv_image = self._zoom_out(cv_image, scale, defect_config[defect].get("background", "mean"))
                labels["zoom_out"] = 1
                metadata["zoom_out_scale"] = float(scale)
            elif defect == "zoom_in":
                crop_scale = self.rng.uniform(*defect_config[defect]["crop_scale_range"])
                cv_image = self._zoom_in(cv_image, crop_scale)
                labels["zoom_in"] = 1
                metadata["zoom_in_crop_scale"] = float(crop_scale)
            elif defect == "color_cast":
                shift = self.rng.uniform(*defect_config[defect]["shift_range"])
                cv_image = self._color_cast(cv_image, shift)
                labels["color_cast"] = 1
                metadata["color_cast_shift"] = float(shift)
            elif defect == "compression":
                quality = self.rng.randint(*defect_config[defect]["quality_range"])
                cv_image = self._jpeg_compress(cv_image, quality)
                labels["compression"] = 1
                metadata["compression_quality"] = float(quality)
            elif defect == "vignette":
                strength = self.rng.uniform(*defect_config[defect]["strength_range"])
                cv_image = self._apply_vignette(cv_image, strength)
                labels["vignette"] = 1
                metadata["vignette_strength"] = float(strength)
            elif defect == "glare":
                radius = self.rng.uniform(*defect_config[defect]["radius_range"])
                glare_intensity = self.rng.uniform(*defect_config[defect]["intensity_range"])
                cv_image = self._add_glare(cv_image, radius, glare_intensity)
                labels["glare"] = 1
                metadata["glare_radius"] = float(radius)
                metadata["glare_intensity"] = float(glare_intensity)
            elif defect == "hair":
                count = self.rng.randint(*defect_config[defect]["count_range"])
                thickness = defect_config[defect]["thickness_range"]
                cv_image = self._add_hair(cv_image, count, tuple(thickness))
                labels["hair"] = 1
                metadata["hair_count"] = float(count)
            applied.append(defect)

        if applied:
            labels["overall_fail"] = 1

        return AugmentationResult(image=self._to_pil(cv_image), labels=labels, metadata=metadata)


__all__ = ["QualityAugmentor", "AugmentationResult"]
