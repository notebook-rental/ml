"""대여 사진 사전 계산과 메모리 캐시 (research R12·R15, tasks T065·T069).

반납 때 기다리는 시간을 줄이려고, 대여(checkout) 사진이 저장되면 그 사진에서 나올 것을 미리 계산해 둔다.

- 노트북 마스크(경량 SAM 또는 GrabCut)
- SIFT 특징점(정렬용)
- DINOv2-S 패치 특징(비교용)
- 비교 해상도로 줄인 사진(JPEG, SSIM·색 비교용)

결과는 `derived/baseline-features/<photoId>.npz`(fp16)로 저장한다. 반납 때는 반납 사진만 계산하면 된다.
`/prefetch`가 오면 노트북 1대의 8컷을 파드 메모리(LRU)에 미리 올려 저장소 읽기 시간도 없앤다.
백본 이름이 저장된 값과 다르면(모델 교체) 다시 계산한다.
"""

from __future__ import annotations

import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from common.messages import QrCode
from common.storage import ObjectNotFound, Storage
from inference.pipeline import align
from inference.pipeline.bank import bank_features
from inference.pipeline.imageutil import decode, encode_jpeg, resize_long_side, to_gray
from inference.pipeline.patch_features import Backbone
from inference.pipeline.segment import Segmenter, laptop_box_from_gray

FORMAT_VERSION = 2  # 2: 조각 맞추기 기억 장소(bank) 추가(docs specs/004). 1이면 다시 계산한다


def cache_key(photo_id) -> str:
    return f"derived/baseline-features/{photo_id}.npz"


@dataclass
class BaselineFeatures:
    photo_id: str
    backbone: str
    segmenter: str
    orig_size: tuple[int, int]  # (w, h) 원본 작업본
    scale: float  # compare 해상도 / 원본
    image: np.ndarray  # compare 해상도 BGR
    mask: np.ndarray  # compare 해상도 노트북 마스크
    keypoints: align.Keypoints
    crop: tuple[int, int, int, int]  # DINOv2에 넣은 크롭(x1,y1,x2,y2), compare 해상도
    dino_size: tuple[int, int]  # DINOv2 입력 (W,H)
    dino: np.ndarray  # [gh, gw, C] float32, 길이 1
    qr: list[QrCode]
    bank: np.ndarray | None = None  # [K, C] float16, 크롭 통째 + 2×2 조각(inference/pipeline/bank.py)

    @property
    def gray(self) -> np.ndarray:
        return to_gray(self.image)

    def nbytes(self) -> int:
        extra = self.bank.nbytes if self.bank is not None else 0
        return self.image.nbytes + self.mask.nbytes + self.keypoints.desc.nbytes + self.dino.nbytes + extra


def mask_crop(mask: np.ndarray, pad: float = 0.03) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    px, py = int(w * pad), int(h * pad)
    return (
        int(max(0, xs.min() - px)),
        int(max(0, ys.min() - py)),
        int(min(w, xs.max() + 1 + px)),
        int(min(h, ys.max() + 1 + py)),
    )


def compute(
    photo_id,
    img: np.ndarray,
    backbone: Backbone,
    segmenter: Segmenter,
    compare_long_side: int,
    qr: list[QrCode] | None = None,
) -> BaselineFeatures:
    h, w = img.shape[:2]
    small, scale = resize_long_side(img, compare_long_side)
    box = laptop_box_from_gray(img)
    mask_full = segmenter.segment(small, None if box is None else tuple(round(v * scale) for v in box))
    kps = align.extract(to_gray(small), mask_full)
    crop = mask_crop(mask_full)
    x1, y1, x2, y2 = crop
    feats = backbone.features(small[y1:y2, x1:x2])
    bank = bank_features(backbone, small[y1:y2, x1:x2])
    return BaselineFeatures(
        photo_id=str(photo_id),
        backbone=backbone.name,
        segmenter=segmenter.name,
        orig_size=(w, h),
        scale=scale,
        image=small,
        mask=mask_full,
        keypoints=kps,
        crop=crop,
        dino_size=backbone.size,
        dino=feats,
        qr=list(qr or []),
        bank=bank,
    )


def serialize(f: BaselineFeatures) -> bytes:
    meta = {
        "format": FORMAT_VERSION,
        "photoId": f.photo_id,
        "backbone": f.backbone,
        "segmenter": f.segmenter,
        "origSize": list(f.orig_size),
        "scale": f.scale,
        "crop": list(f.crop),
        "dinoSize": list(f.dino_size),
        "qr": [q.model_dump() for q in f.qr],
    }
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
        image=np.frombuffer(encode_jpeg(f.image, 92), np.uint8),
        mask=f.mask,
        kp_pts=f.keypoints.pts.astype(np.float32),
        kp_desc=f.keypoints.desc.astype(np.float16),
        dino=f.dino.astype(np.float16),
        bank=(f.bank if f.bank is not None else np.zeros((0, f.dino.shape[-1]))).astype(np.float16),
    )
    return buf.getvalue()


def deserialize(data: bytes) -> BaselineFeatures:
    z = np.load(io.BytesIO(data), allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode())
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError("지원하지 않는 기준 특징 형식")
    return BaselineFeatures(
        photo_id=meta["photoId"],
        backbone=meta["backbone"],
        segmenter=meta["segmenter"],
        orig_size=tuple(meta["origSize"]),
        scale=float(meta["scale"]),
        image=decode(bytes(z["image"])),
        mask=z["mask"],
        keypoints=align.Keypoints(z["kp_pts"], z["kp_desc"].astype(np.float32)),
        crop=tuple(meta["crop"]),
        dino_size=tuple(meta["dinoSize"]),
        dino=z["dino"].astype(np.float32),
        qr=[QrCode.model_validate(q) for q in meta["qr"]],
        bank=z["bank"],
    )


class BaselineStore:
    """기준 특징 읽기: 메모리 LRU → 저장소 npz → (없으면) 그 자리에서 계산 후 저장."""

    def __init__(self, storage: Storage, capacity: int):
        self.storage = storage
        self.capacity = capacity
        self._lru: OrderedDict[str, BaselineFeatures] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _remember(self, f: BaselineFeatures) -> None:
        with self._lock:
            self._lru[f.photo_id] = f
            self._lru.move_to_end(f.photo_id)
            while len(self._lru) > self.capacity:
                self._lru.popitem(last=False)

    def cached(self, photo_id) -> BaselineFeatures | None:
        with self._lock:
            f = self._lru.get(str(photo_id))
            if f is not None:
                self._lru.move_to_end(str(photo_id))
            return f

    def load(self, photo_id, backbone_name: str) -> BaselineFeatures | None:
        """메모리 또는 저장소에서. 없거나 백본이 다르면 None."""
        f = self.cached(photo_id)
        if f is not None and f.backbone == backbone_name:
            self.hits += 1
            return f
        self.misses += 1
        try:
            f = deserialize(self.storage.get_bytes(cache_key(photo_id)))
        except (ObjectNotFound, ValueError, KeyError):
            return None
        if f.backbone != backbone_name:
            return None
        self._remember(f)
        return f

    def save(self, f: BaselineFeatures) -> BaselineFeatures:
        """저장하고, 저장된 형태(JPEG·fp16)로 다시 읽은 값을 돌려준다.

        처음 계산한 값과 저장소에서 읽은 값이 조금이라도 다르면 같은 비교가 실행마다 다른 결과를 낼 수 있다(멱등성 깨짐).
        그래서 처음부터 저장된 형태를 쓴다.
        """
        data = serialize(f)
        self.storage.put_bytes(cache_key(f.photo_id), data)
        f = deserialize(data)
        self._remember(f)
        return f

    def get_or_compute(
        self,
        photo_id,
        image_key: str,
        backbone: Backbone,
        segmenter: Segmenter,
        compare_long_side: int,
        qr: list[QrCode] | None = None,
    ) -> BaselineFeatures:
        f = self.load(photo_id, backbone.name)
        if f is not None:
            return f
        img = decode(self.storage.get_bytes(image_key))
        return self.save(compute(photo_id, img, backbone, segmenter, compare_long_side, qr))


def warp_mask_to_target(mask: np.ndarray, H_t2b: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """기준 마스크를 반납 사진 좌표로 옮긴다(반납 사진은 SAM을 다시 돌리지 않는다, R12)."""
    h, w = target_shape
    return cv2.warpPerspective(mask, np.linalg.inv(H_t2b), (w, h), flags=cv2.INTER_NEAREST)
