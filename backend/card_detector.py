"""Modulo de deteccion de cartas de Yu-Gi-Oh! en imagenes completas (OpenCV).

Pipeline robusto de Vision por Computadora:
1. Filtro bilateral para preservar bordes nitidos de cartas suavizando texturas y ruido.
2. Deteccion de bordes multi-umbral: Canny de rango amplio + Adaptive Thresholding (Otsu & Gaussian).
3. Busqueda de contornos exhaustiva (RETR_LIST) para detectar cartas dentro de tapetes y zonas de juego.
4. Envolvente convexa (cv2.convexHull) sobre cada contorno para unificar el contorno exterior
   y sellar textos o ilustraciones interiores.
5. Filtro geometrico por relacion de aspecto de carta (~0.686 en vertical / ~1.459 en horizontal)
   con tolerancia para perspectiva inclinada.
6. Supresion de no-maximos (NMS) por superposicion y cercania de centros.
7. Rectificacion proyectiva (4-point perspective transform / dewarping) a resolucion canonica (268x391 px).
"""

from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from PIL import Image

CARD_TARGET_WIDTH = 268
CARD_TARGET_HEIGHT = 391
CARD_ASPECT_RATIO = CARD_TARGET_WIDTH / CARD_TARGET_HEIGHT  # ~0.6854


@dataclass
class DetectedCard:
    """Representa una carta detectada y recortada en el encuadre."""
    crop: Image.Image
    box: dict[str, Any]
    points: list[list[int]]
    is_defense_mode: bool
    contour_area: float


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos 2D en sentido horario: [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left tiene la menor suma (x + y), bottom-right tiene la mayor suma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right tiene la menor diferencia (y - x), bottom-left la mayor diferencia
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


class CardDetector:
    """Detector de cartas de Yu-Gi-Oh! optimizado para camaras reales y mesas de juego."""

    def __init__(
        self,
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.40,
        aspect_min: float = 0.45,
        aspect_max: float = 0.90,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

    def detect(self, pil_image: Image.Image) -> list[DetectedCard]:
        """Detecta todas las cartas visibles en la imagen y devuelve sus recortes alineados."""
        rgb_img = pil_image.convert("RGB")
        img_np = np.array(rgb_img)
        h_orig, w_orig = img_np.shape[:2]
        total_area = float(h_orig * w_orig)

        # 1. Preprocesamiento: bilateral filter preserva los bordes rectos de las cartas
        # mientras elimina el grano y texturas finas de maderas o tapetes
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)

        # 2. Multiples tecnicas de binarizacion complementarias
        # A) Canny para bordes con contraste
        edges_canny = cv2.Canny(blurred, 25, 95)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel3)

        # B) Umbral adaptativo para escenas con sombras o iluminacion no uniforme
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 4
        )
        closed_adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel3)

        # C) Otsu thresholding para fondos de color uniforme
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        closed_otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel3)

        # 3. Extraer contornos combinados con RETR_LIST (no omite cartas dentro de tapetes)
        all_contours: list[np.ndarray] = []
        for mask in [closed_canny, closed_adapt, closed_otsu]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(cnts)

        candidates = []

        for cnt in all_contours:
            # Envolvente convexa: esencial para unificar el perimetro de la carta
            # sin que se fragmente por ilustraciones o cuadros de texto internos
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)

            # Rango de tamano razonable para cartas en mesa
            if area < (self.min_area_ratio * total_area) or area > (self.max_area_ratio * total_area):
                continue

            rect = cv2.minAreaRect(hull)
            rw, rh = rect[1]
            if rw < 25 or rh < 25:
                continue

            rect_area = rw * rh
            solidity = area / rect_area
            # Las cartas son casi rectangulos perfectos (> 70% de solidez)
            if solidity < 0.70:
                continue

            aspect = min(rw, rh) / max(rw, rh)
            # Rango de aspect ratio admitiendo deformacion de perspectiva
            if not (self.aspect_min <= aspect <= self.aspect_max):
                continue

            box = cv2.boxPoints(rect)
            candidates.append((rect, area, box))

        # 4. Supresion de no-maximos por distancia de centros y solapamiento
        candidates.sort(key=lambda x: x[1], reverse=True)
        filtered = []

        for c in candidates:
            center = c[0][0]
            center_dist_min = max(c[0][1]) * 0.35  # Distancia minima proporcional
            is_dup = False

            for ex in filtered:
                ex_center = ex[0][0]
                dist = np.hypot(center[0] - ex_center[0], center[1] - ex_center[1])
                if dist < max(45.0, center_dist_min):
                    is_dup = True
                    break

            if not is_dup:
                filtered.append(c)

        # 5. Dewarping y generacion de crops canónicos (268x391)
        results: list[DetectedCard] = []

        for rect, area, box in filtered:
            pts = box.astype(np.float32)
            ordered_pts = order_quad_points(pts)

            tl, tr, br, bl = ordered_pts
            width_top = np.linalg.norm(tr - tl)
            width_bot = np.linalg.norm(br - bl)
            width = max(width_top, width_bot)

            height_right = np.linalg.norm(br - tr)
            height_left = np.linalg.norm(bl - tl)
            height = max(height_right, height_left)

            # Si el ancho aparente es mayor que el alto, la carta esta en modo defensa (horizontal)
            is_defense = bool(width > height)

            if not is_defense:
                dst_pts = np.array([
                    [0, 0],
                    [CARD_TARGET_WIDTH - 1, 0],
                    [CARD_TARGET_WIDTH - 1, CARD_TARGET_HEIGHT - 1],
                    [0, CARD_TARGET_HEIGHT - 1]
                ], dtype="float32")
            else:
                # Rotar 90 grados al dewarpear para que llegue orientada verticalmente al clasificador
                dst_pts = np.array([
                    [0, CARD_TARGET_HEIGHT - 1],
                    [0, 0],
                    [CARD_TARGET_WIDTH - 1, 0],
                    [CARD_TARGET_WIDTH - 1, CARD_TARGET_HEIGHT - 1]
                ], dtype="float32")

            M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            warped = cv2.warpPerspective(
                img_np,
                M,
                (CARD_TARGET_WIDTH, CARD_TARGET_HEIGHT),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )

            x_min = int(max(0, np.min(pts[:, 0])))
            y_min = int(max(0, np.min(pts[:, 1])))
            x_max = int(min(w_orig, np.max(pts[:, 0])))
            y_max = int(min(h_orig, np.max(pts[:, 1])))

            box_dict = {
                "x": x_min,
                "y": y_min,
                "width": max(1, x_max - x_min),
                "height": max(1, y_max - y_min),
            }

            points_list = [[int(p[0]), int(p[1])] for p in ordered_pts]
            crop_pil = Image.fromarray(warped)

            results.append(
                DetectedCard(
                    crop=crop_pil,
                    box=box_dict,
                    points=points_list,
                    is_defense_mode=bool(is_defense),
                    contour_area=float(area),
                )
            )

        return results
