"""기존 손상 겹침 판정 (tasks T062).

이미 등록된 손상(device_damage)과 같은 자리의 후보는 "새 손상"이 아닐 가능성이 크다.
후보 면적의 30% 이상이 기존 손상 영역과 겹치면 overlapsDamageId에 그 손상 ID를 넣는다.
직원 화면은 이 표시를 보고 "기존 손상"으로 빠르게 판정할 수 있다(최종 판정은 사람, 헌법 I).

기존 손상 좌표는 기준(대여) 사진 원본 픽셀 좌표로 받는다(device_damage는 기준 사진 좌표계에 저장된다).
"""

from __future__ import annotations

import cv2
import numpy as np

from common.messages import ExistingDamage

OVERLAP_RATIO = 0.3


def damage_mask(shape: tuple[int, int], damage: ExistingDamage, scale: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    coords = damage.geometry.coords
    if damage.geometry.type == "box" and len(coords) == 2:
        (x1, y1), (x2, y2) = coords
        coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if len(coords) < 3:
        return mask
    pts = (np.array(coords, np.float32) * scale).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def overlapping_damage(region_mask: np.ndarray, damages: list[ExistingDamage], scale: float):
    """가장 많이 겹치는 기존 손상의 ID(겹침 비율 OVERLAP_RATIO 이상일 때), 없으면 None."""
    area = int((region_mask > 0).sum())
    if area == 0:
        return None
    best_id, best = None, 0.0
    for d in damages:
        dm = damage_mask(region_mask.shape, d, scale)
        ratio = float(((region_mask > 0) & (dm > 0)).sum()) / area
        if ratio > best:
            best_id, best = d.id, ratio
    return best_id if best >= OVERLAP_RATIO else None
