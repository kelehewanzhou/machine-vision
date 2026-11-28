"""
crop.py —— 候选取景框生成与裁剪模块（接口规范 v1.0）

公共接口
--------
    generate_candidates(image_path, image_id, config, *, group_id) -> Result[CandidateBatch]

    image_path : 项目根目录相对路径，JPEG / PNG
    返回 data  : {"candidates": [Candidate, ...],
                    "statistics": {"proposed", "filtered", "deduplicated",
                                "write_failed", "returned"}}

处理流程
--------
    0. 解码：EXIF 方向校正 → 透明 PNG 白底合成 → RGB
    1. 提案：比例 → 尺度 → 横位置 → 纵位置，宽高与左上角取 floor；原图 crop_id=0
    2. 精确重复框去除（保留最先出现项）
    3. 主体定位：YuNet 人脸并集框扩大；无人脸时用谱残差显著性主体框
    4. 约束过滤：主体覆盖率 ≥ 0.90 且显著性保留率 ≥ 0.70（值为 null 时该项不生效）
    5. IoU 去重：按提案顺序与已保留项比较，IoU ≥ 0.85 视为重复
    6. 写出裁剪 PNG：outputs/{run_id}/crops/{group_id}/{image_id}/{crop_id}.png

坐标统一为 [x1, y1, x2, y2]，整数、左闭右开，对应 EXIF 校正后的原图。
"""

from __future__ import annotations

from PIL import Image, ImageOps
from pathlib import Path
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from fractions import Fraction
from math import floor, ceil
import math

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
@dataclass
class CropConfig:
    # 候选框生成
    scales: Sequence[float] = (0.8, 0.9, 1.0)
    # 长宽比 = 宽 / 高
    aspects: Dict[str, Fraction] = field(
        default_factory=lambda: {
            "1:1": Fraction(1, 1),
            "4:5": Fraction(4, 5),
            "16:9": Fraction(16, 9)
        }
    )
    # 3×3 位置：0 = 靠左/靠上，0.5 = 居中，1 = 靠右/靠下
    positions: Sequence[float] = (0.0, 0.5, 1.0)

    # 约束过滤阈值
    subject_coverage: float = 0.90   # 主体框落在候选框内的面积比例下限
    keep_threshold: float = 0.70     # 显著性保留率下限
    iou_threshold: float = 0.85      # 去重 IoU 阈值

    # 显著性主体框
    saliency_width: int = 64         # 谱残差计算时的缩放宽度
    saliency_top: float = 0.20       # 取显著性最高的前 20% 像素
    close_kernel_ratio: float = 0.05 # 闭运算核边长 = 短边 × 该比例

    # 人脸保护框（相对人脸并集框的宽 / 高）
    face_expand_left: float = 0.35
    face_expand_right: float = 0.35
    face_expand_top: float = 0.50
    face_expand_bottom: float = 0.20
    face_score_threshold: float = 0.80

    face_model_path: Optional[str] = "models/face_detection_yunet_2023mar.onnx"
    face_max_side: int = 1280

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CropConfig":
        """读取 config["crop"]；没写的用默认值，非法抛 ValueError。"""
        d = {} if d is None else d
        if not isinstance(d, dict):
            raise ValueError("crop 配置必须为字典")

        cfg = cls()

        def is_num(x):
            return (isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x))

        def num_list(key, lo, hi, lo_open):
            v = d[key]
            ok = isinstance(v, list) and v and all(
                is_num(x) and (lo < x if lo_open else lo <= x) and x <= hi for x in v)
            if not ok:
                raise ValueError(f"crop.{key} 取值非法")
            return tuple(float(x) for x in v)

        # 键名 → 处理方式（键名等组长的 default.json 定了再对齐）
        known = {"aspect_ratios", "scales", "positions", "iou_threshold",
                 "subject_coverage_min", "saliency_retention_min",
                 "face_model_path", "face_score_threshold", "face_max_side"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"crop 配置含未知键：{sorted(unknown)}")

        if "aspect_ratios" in d:
            ar = d["aspect_ratios"]
            if not (isinstance(ar, list) and ar and all(
                    isinstance(r, list) and len(r) == 2 and
                    all(type(v) is int and v > 0 for v in r) for r in ar)):
                raise ValueError("crop.aspect_ratios 每项须为两个正整数")
            cfg.aspects = {f"{a}:{b}": Fraction(a, b) for a, b in ar}
        if "scales" in d:
            cfg.scales = num_list("scales", 0, 1, lo_open=True)
        if "positions" in d:
            cfg.positions = num_list("positions", 0, 1, lo_open=False)

        if "iou_threshold" in d:
            v = d["iou_threshold"]
            if not (is_num(v) and 0 < v <= 1):
                raise ValueError("crop.iou_threshold 须在 (0,1]")
            cfg.iou_threshold = float(v)

        for key, attr in (("subject_coverage_min", "subject_coverage"),
                            ("saliency_retention_min", "keep_threshold"),
                            ("face_score_threshold", "face_score_threshold")):
            if key in d:
                v = d[key]
                if not (is_num(v) and 0 <= v <= 1):
                    raise ValueError(f"crop.{key} 须在 [0,1]")
                setattr(cfg, attr, float(v))

        if "face_model_path" in d:
            v = d["face_model_path"]
            if v is not None and not (isinstance(v, str) and v):
                raise ValueError("crop.face_model_path 须为 null 或非空字符串")
            if v is not None:
                resolve_project_path(v)      # 非法路径会抛 ValueError
            cfg.face_model_path = v
        if "face_max_side" in d:
            v = d["face_max_side"]
            if not (type(v) is int and v > 0):
                raise ValueError("crop.face_max_side 须为正整数")
            cfg.face_max_side = v

        return cfg


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def resolve_project_path(relative_path: str) -> Path:
    root = Path(__file__).resolve().parents[1]

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("路径必须为非空字符串")

    if "\\" in relative_path or ":" in relative_path or "\x00" in relative_path:
        raise ValueError("不合法的项目路径")

    path = Path(relative_path)

    if path.is_absolute() or path.drive or path.root:
        raise ValueError("必须使用项目相对路径")

    if ".." in path.parts:
        raise ValueError("不允许访问上级目录")

    target = (root / path).resolve()

    if not target.is_relative_to(root):
        raise ValueError("路径超出项目目录")

    return target

def validate_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("ID只能包含字母、数字、下划线和连字符")

class CropImageError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        
class CropModelError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

def read_image(path: str, max_pixels: int = 40_000_000) -> np.ndarray:
    with Image.open(path) as img:
        if img.format not in ("JPEG", "PNG"):
            raise CropImageError(
                "UNSUPPORTED_FORMAT", "不支持的图片格式"
                )

        width, height = img.size

        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise CropImageError(
                "INVALID_IMAGE_SIZE", "图片尺寸不符合要求"
                )

        img = ImageOps.exif_transpose(img)

        if "A" in img.getbands() or "transparency" in img.info:
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba)

        img = img.convert("RGB")

        return np.array(img)

def box_area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a: Box, b: Box) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0

def clip_box(b: Sequence[float], width: int, height: int) -> Box:
    """限制候选框坐标，确保矩形框不超出原图边界"""
    x1 = floor(min(max(b[0], 0), width))
    y1 = floor(min(max(b[1], 0), height))
    x2 = ceil(min(max(b[2], 0), width))
    y2 = ceil(min(max(b[3], 0), height))
    return (x1, y1, x2, y2)


# ----------------------------------------------------------------------------
# 1. 候选框生成
# ----------------------------------------------------------------------------
def largest_box_size(width: int, height: int, ratio: Fraction) -> Tuple[Fraction, Fraction]:
    """指定长宽比下的最大内接矩形"""
    w = Fraction(width)
    h = Fraction(height)

    if w / h > ratio:
        return h * ratio, h

    return w, w / ratio


def generate_boxes(width: int, height: int, cfg: CropConfig) -> List[dict]:
    full = (0, 0, width, height)
    out = [{
        "id": 0,
        "box": full,
        "scale": 1.0,
        "aspect": "original",
        "position": None
    }]
    seen = {full}
    next_id = 1

    for aspect_name, ratio in cfg.aspects.items():
        base_w, base_h = largest_box_size(width, height, ratio)

        for s in cfg.scales:
            scale = Fraction(str(s))

            cw = max(1, min(width, floor(base_w * scale)))
            ch = max(1, min(height, floor(base_h * scale)))

            for px in cfg.positions:
                for py in cfg.positions:
                    x1 = floor((width - cw) * Fraction(str(px)))
                    y1 = floor((height - ch) * Fraction(str(py)))

                    b = (x1, y1, x1 + cw, y1 + ch)

                    if b in seen:
                        continue

                    seen.add(b)

                    out.append({
                        "id": next_id,
                        "box": b,
                        "scale": float(s),
                        "aspect": aspect_name,
                        "position": [float(px), float(py)]
                    })

                    next_id += 1

    return out


# ----------------------------------------------------------------------------
# 2a. 谱残差显著性 + 自动主体框
# ----------------------------------------------------------------------------
def spectral_residual_saliency(
    image: np.ndarray, small_width: int = 64
) -> Optional[np.ndarray]:

    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    if gray.max() == gray.min():
        return None

    scale = small_width / max(h, w)
    
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))

    small = cv2.resize(
        gray, (sw, sh), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    if small.max() == small.min():
        return None

    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)

    residual = log_amp - cv2.blur(log_amp, (3, 3))

    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=2.5)

    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)

    if not np.isfinite(sal).all():
        return None

    sal -= sal.min()

    if sal.max() < 1e-8:
        return None

    sal /= sal.max()

    return sal


def saliency_subject_box(
    saliency: Optional[np.ndarray], cfg: CropConfig
) -> Optional[Box]:

    if saliency is None:
        return None

    h, w = saliency.shape
    if not np.isfinite(saliency).all() or float(saliency.max() - saliency.min()) < 1e-8:
        return None   # 纯色图等无有效显著区域的情况
    thr = np.quantile(saliency, 1.0 - cfg.saliency_top)
    mask = (saliency >= thr).astype(np.uint8)
    k = max(3, int(round(min(h, w) * cfg.close_kernel_ratio)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[i, :4]
    return (int(x), int(y), int(x + bw), int(y + bh))


# ----------------------------------------------------------------------------
# 2b. 人脸保护框（每张图重新检测，不复用上一张的结果）
# ----------------------------------------------------------------------------
_DETECTORS: Dict[str, "cv2.FaceDetectorYN"] = {}

def find_yunet_model(model_path: Optional[str] = None) -> Optional[str]:
    relative = model_path or "models/face_detection_yunet_2023mar.onnx"
    path = resolve_project_path(relative)
    return str(path) if path.is_file() else None


def detect_faces(
    image: np.ndarray,
    model_path: Optional[str],
    cfg: CropConfig
) -> List[Box]:

    path = find_yunet_model(model_path)

    if path is None:
        raise CropModelError(
            "MODEL_UNAVAILABLE", "YuNet模型不存在"
        )

    h, w = image.shape[:2]

    scale = min(1.0, cfg.face_max_side / max(h, w))
    
    dw = max(1, round(w * scale))
    dh = max(1, round(h * scale))

    if scale < 1.0:
        small = cv2.resize(
            image, (dw, dh), interpolation=cv2.INTER_AREA
        )
    else:
        small = image

    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    det = _DETECTORS.get(path)

    if det is None:
        try:
            det = cv2.FaceDetectorYN.create(
                path, "", (dw, dh),
                cfg.face_score_threshold, 0.3, 5000
            )

        except (cv2.error, OSError) as e:
            raise CropModelError(
                "MODEL_UNAVAILABLE", "YuNet模型初始化失败"
            ) from e

        _DETECTORS[path] = det

    try:
        det.setInputSize((dw, dh))
        det.setScoreThreshold(cfg.face_score_threshold)

        _, faces = det.detect(bgr)

    except cv2.error as e:
        raise CropModelError(
            "INFERENCE_FAILED", "YuNet人脸检测失败"
        ) from e

    if faces is None:
        return []

    sx = w / dw
    sy = h / dh

    boxes = []

    for f in faces:
        x, y, fw, fh = map(float, f[:4])
        
        if not all(math.isfinite(v) for v in (x, y, fw, fh)):
            continue

        if fw <= 0 or fh <= 0:
            continue

        b = clip_box(
            (x * sx, y * sy,
             (x + fw) * sx, (y + fh) * sy),
            w, h
        )

        if box_area(b) > 0:
            boxes.append(b)

    return boxes


def face_protect_box(faces: List[Box], width: int, height: int, cfg: CropConfig) -> Optional[Box]:
    """多张人脸取并集外接框，再按 左右35% / 上50% / 下20% 扩大。"""
    if not faces:
        return None
    x1 = min(f[0] for f in faces); y1 = min(f[1] for f in faces)
    x2 = max(f[2] for f in faces); y2 = max(f[3] for f in faces)
    fw, fh = x2 - x1, y2 - y1
    return clip_box((x1 - cfg.face_expand_left * fw,
                     y1 - cfg.face_expand_top * fh,
                     x2 + cfg.face_expand_right * fw,
                     y2 + cfg.face_expand_bottom * fh), width, height)


# ----------------------------------------------------------------------------
# 3. 约束过滤 + 4. IoU 去重
# ----------------------------------------------------------------------------
def keep_ratio(saliency_integral: Optional[np.ndarray], b: Box) -> Optional[float]:
    if saliency_integral is None:
        return None

    ii = saliency_integral
    total = ii[-1, -1]

    if not np.isfinite(total) or total <= 0:
        return None

    x1, y1, x2, y2 = b
    s = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]

    return float(np.clip(s / total, 0.0, 1.0))


def subject_coverage(subject: Optional[Box], b: Box) -> Optional[float]:
    if subject is None or box_area(subject) == 0:
        return None

    return intersection(subject, b) / box_area(subject)


def nms_dedup(cands: List[dict], thr: float) -> List[dict]:
    full = [c for c in cands if c["id"] == 0]
    rest = [c for c in cands if c["id"] != 0]
    kept = list(full)

    for c in rest:
        if all(iou(c["box"], k["box"]) < thr for k in kept):
            kept.append(c)

    return kept


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def process_image(
    image: np.ndarray,
    config: Optional[CropConfig] = None,
    model_path: Optional[str] = None,
    mode: str = "best_overall"
):
    cfg = config or CropConfig()
    h, w = image.shape[:2]
    
    if mode not in ("best_original", "best_overall"):
        raise ValueError("不支持的运行模式")

    if mode == "best_original":
        original = {
            "id": 0,
            "box": [0, 0, w, h],
            "scale": 1.0,
            "aspect": "original",
            "position": None,
            "keep_ratio": None,
            "coverage": None
        }

        details = {
            "image_size": (w, h),
            "saliency": None,
            "faces": [],
            "subject_box": None,
            "subject_method": "none",
            "generated": [original],
            "filtered": [original],
            "final": [original],
            "counts": {
                "generated": 1,
                "after_constraints": 1,
                "after_iou": 1
            },
            "statistics": {
                "proposed": 1,
                "filtered": 0,
                "deduplicated": 0,
                "write_failed": 0,
                "returned": 1
            }
        }

        return [original], details

    saliency_error = None

    try:
        sal = spectral_residual_saliency(image, cfg.saliency_width)

        integral = (
            None if sal is None
            else cv2.integral(sal.astype(np.float64))
        )

    except (cv2.error, CropModelError):
        sal = None
        integral = None

        saliency_error = {
            "code": "INFERENCE_FAILED",
            "message": "显著性计算发生异常，已跳过显著性约束"
        }

    model_error = None
    faces = []
    if model_path is not None:
        try:
            faces = detect_faces(image, model_path, cfg)
        except CropModelError as e:
            model_error = {"code": e.code,
                            "message": str(e) + "，已降级为显著性检测"}

    if faces:
        subject = face_protect_box(faces, w, h, cfg)
        method = "face"
    else:
        subject = saliency_subject_box(sal, cfg)
        method = "saliency" if subject is not None else "none"

    generated = generate_boxes(w, h, cfg)

    for c in generated:
        c["keep_ratio"] = keep_ratio(integral, c["box"])
        c["coverage"] = subject_coverage(subject, c["box"])

    if subject is None:
        # 策略 B：没有主体依据就不裁剪，只保留原图（被排除的裁剪计入 filtered）
        filtered = [c for c in generated if c["id"] == 0]
    else:
        filtered = [
            c for c in generated
            if c["id"] == 0 or (
                (c["coverage"] is None or
                c["coverage"] >= cfg.subject_coverage)
                and
                (c["keep_ratio"] is None or
                c["keep_ratio"] >= cfg.keep_threshold)
            )
        ]

    final = nms_dedup(filtered, cfg.iou_threshold)
    proposed = (
        1 + len(cfg.aspects)
        * len(cfg.scales)
        * len(cfg.positions) ** 2
    )

    exact_duplicates = proposed - len(generated)
    iou_duplicates = len(filtered) - len(final)

    statistics = {
        "proposed": proposed,
        "filtered": len(generated) - len(filtered),
        "deduplicated": exact_duplicates + iou_duplicates,
        "write_failed": 0,
        "returned": len(final)
    }

    candidates = [
        {
            "id": c["id"],
            "box": list(c["box"]),
            "scale": c["scale"],
            "aspect": c["aspect"],
            "position": c["position"],
            "keep_ratio": c["keep_ratio"],
            "coverage": c["coverage"]
        }
        for c in final
    ]

    details = {
        "image_size": (w, h),
        "saliency": sal,
        "faces": faces,
        "subject_box": subject,
        "subject_method": method,
        "generated": generated,
        "filtered": filtered,
        "final": final,
        "statistics": statistics,
        "model_error": model_error,
        "saliency_error": saliency_error,
        "counts": {
            "generated": len(generated),
            "after_constraints": len(filtered),
            "after_iou": len(final)
        }
    }

    return candidates, details

def save_crop(
    image: np.ndarray,
    box: Box,
    run_id: str,
    group_id: str,
    image_id: str,
    crop_id: int
) -> str:

    for value in (run_id, group_id, image_id):
        validate_id(value)

    if type(crop_id) is not int or crop_id <= 0:
        raise ValueError("裁剪ID必须为正整数")

    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("候选框坐标不合法")

    relative_path = (
        f"outputs/{run_id}/crops/"
        f"{group_id}/{image_id}/{crop_id}.png"
    )

    path = resolve_project_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cropped = image[y1:y2, x1:x2]

    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".png", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            Image.fromarray(np.ascontiguousarray(cropped)).save(
                f, format="PNG", compress_level=1
            )
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return relative_path

def build_candidate(
    c: dict,
    group_id: str,
    image_id: str,
    image_path: str,
    crop_path: str,
    width: int,
    height: int,
    details: dict
) -> dict:

    subject = details["subject_box"]

    return {
        "group_id": group_id,
        "image_id": image_id,
        "image_path": image_path,
        "crop_id": c["id"],
        "box": list(c["box"]),
        "image_width": width,
        "image_height": height,
        "crop_path": crop_path,
        "is_original": c["id"] == 0,
        "subject_coverage": c["coverage"],
        "saliency_retention": c["keep_ratio"],
        "metadata": {
            "scale": c["scale"],
            "aspect": c["aspect"],
            "position": c["position"],
            "subject_box": None if subject is None else list(subject),
            "subject_method": details["subject_method"]
        }
    }
    
def make_result(
    status: str,
    reason: str,
    data=None,
    errors=None
) -> dict:

    return {
        "schema_version": "1.0",
        "status": status,
        "reason": reason,
        "errors": [] if errors is None else errors,
        "data": data
    }
    
def make_error(
    code: str,
    message: str,
    context=None
) -> dict:

    return {
        "code": code,
        "module": "crop",
        "message": message,
        "retryable": False,
        "context": {} if context is None else context
    }

def generate_candidates(
    image_path: str,
    image_id: str,
    config: dict,
    *,
    group_id: str
) -> dict:

    context = {}

    try:
        validate_id(group_id)
        context["group_id"] = group_id

        validate_id(image_id)
        context["image_id"] = image_id

        path = resolve_project_path(image_path)

    except (ValueError, OSError) as e:
        error = make_error("INVALID_INPUT", str(e), context)
        return make_result("error", str(e), errors=[error])

    try:
        if not isinstance(config, dict):
            raise ValueError("config必须为字典")

        mode = config.get("mode", "best_original")

        if mode not in ("best_original", "best_overall"):
            raise ValueError("不支持的运行模式")

        runtime = config.get("runtime")

        if not isinstance(runtime, dict):
            raise ValueError("缺少runtime配置")

        run_id = runtime.get("run_id")
        validate_id(run_id)

        image_cfg = config.get("image", {})

        if not isinstance(image_cfg, dict):
            raise ValueError("image配置必须为字典")

        max_pixels = image_cfg.get(
            "max_image_pixels", 40_000_000
        )

        if type(max_pixels) is not int or max_pixels <= 0:
            raise ValueError("最大像素数必须为正整数")
        cfg = CropConfig.from_dict(config.get("crop"))     

    except ValueError as e:
        error = make_error("INVALID_CONFIG", str(e), context)
        return make_result("error", str(e), errors=[error])

    try:
        image = read_image(str(path), max_pixels)

    except FileNotFoundError:
        error = make_error(
            "IMAGE_NOT_FOUND", "图片文件不存在", context
        )
        return make_result("error", error["message"], errors=[error])

    except CropImageError as e:
        error = make_error(e.code, str(e), context)
        return make_result("error", error["message"], errors=[error])

    except Image.DecompressionBombError:
        error = make_error(
            "INVALID_IMAGE_SIZE", "图片像素数量过大", context
        )
        return make_result("error", error["message"], errors=[error])

    except OSError:
        error = make_error(
            "IMAGE_DECODE_FAILED", "图片无法读取或解码", context
        )
        return make_result("error", error["message"], errors=[error])

    h, w = image.shape[:2]

    candidates, details = process_image(
        image, cfg, model_path=cfg.face_model_path, mode=mode
    )

    records = []
    errors = []
    statistics = details["statistics"].copy()

    for c in candidates:
        if c["id"] == 0:
            crop_path = image_path
        else:
            try:
                crop_path = save_crop(
                    image,
                    c["box"],
                    run_id,
                    group_id,
                    image_id,
                    c["id"]
                )

            except OSError:
                statistics["write_failed"] += 1

                errors.append(make_error(
                    "CROP_WRITE_FAILED",
                    "裁剪图片保存失败",
                    {**context, "crop_id": c["id"]}
                ))

                continue

        record = build_candidate(
            c,
            group_id,
            image_id,
            image_path,
            crop_path,
            w,
            h,
            details
        )

        records.append(record)

    statistics["returned"] = len(records)

    data = {
        "candidates": records,
        "statistics": statistics
    }

    model_error = details.get("model_error")

    if model_error is not None:
        errors.append(make_error(
            model_error["code"],
            model_error["message"],
            context
        ))
    if mode == "best_overall" and details["subject_box"] is None:
        errors.append(make_error(
            "INFERENCE_FAILED",
            "未定位到主体，按策略仅返回原图",
            {**context, "dimension": "subject_coverage"}
        ))
    
    saliency_error = details.get("saliency_error")

    if saliency_error is not None:
        errors.append(make_error(
            saliency_error["code"],
            saliency_error["message"],
            {**context, "dimension": "saliency_retention"}
        ))
        
    if errors:
        return make_result(
            "partial",
            "候选生成完成，部分功能发生降级或失败",
            data,
            errors=errors
        )

    return make_result("ok", "候选生成完成", data)

# ----------------------------------------------------------------------------
# 可视化（实验用）
# ----------------------------------------------------------------------------
def draw_boxes(image: np.ndarray, details: dict, show: str = "final") -> np.ndarray:
    """在 RGB 原图上画候选框（绿）、主体框（红）、人脸（黄），可直接 plt.imshow。"""
    vis = image.copy()
    t = max(2, int(round(min(image.shape[:2]) / 250)))

    def rect(b, color, thick):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), color, thick)   # 左闭右开 → 右下减 1

    for c in details[show]:
        if c["id"] != 0:
            rect(c["box"], (80, 200, 80), t)
    if details["subject_box"]:
        rect(details["subject_box"], (230, 40, 40), t + 1)
    for f in details["faces"]:
        rect(f, (255, 220, 0), t)
    return vis


def crop(image: np.ndarray, b: Box) -> np.ndarray:
    """按内部格式 (x1, y1, x2, y2) 裁剪。"""
    x1, y1, x2, y2 = b
    return image[y1:y2, x1:x2]

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 4:
        print("用法: python src/crop.py 图片相对路径 image_id group_id [run_id]")
        sys.exit(1)

    cfg = {"mode": "best_overall",
            "runtime": {"run_id": sys.argv[4] if len(sys.argv) > 4 else "debug_crop"}}
    res = generate_candidates(sys.argv[1], sys.argv[2], cfg, group_id=sys.argv[3])
    print(json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False))
