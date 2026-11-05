"""자산 번호 라벨(QR·바코드) 검출(spec 002)."""

import numpy as np

from inference.pipeline.qr import detect_barcodes, detect_labels
from tests.fixtures import synth


def _scene_with_barcode(value: str) -> np.ndarray:
    img = synth.base_scene(0).image.copy()  # 기본 장면에는 QR이 하나 있다
    bc = synth.barcode_image(value)
    h, w = bc.shape[:2]
    img[900 : 900 + h, 200 : 200 + w] = bc
    return img


def test_detect_barcodes_reads_code128_with_corners():
    img = _scene_with_barcode("A-001")
    found = detect_barcodes(img)
    assert [b.value for b in found] == ["A-001"]
    xs = [c[0] for c in found[0].corners]
    assert min(xs) >= 150 and max(xs) <= 200 + 500


def test_detect_labels_merges_qr_and_barcode():
    values = {q.value for q in detect_labels(_scene_with_barcode("A-001"))}
    assert values == {synth.QR_VALUE, "A-001"}


def test_detect_barcodes_empty_on_blank():
    assert detect_barcodes(np.zeros((200, 300, 3), np.uint8)) == []
