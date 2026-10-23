#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAI-Net-Signal: 从已验证的 metadata 导出 YOLO 检测标签（v1.0）。

输入：signal_dataset_bbox_measured/metadata.json 与对应 clean / low_snr PNG。
输出：bbox_yolo_export/low_snr 和 /clean 两套 YOLO 数据视图、叠框预览和审计报告。
不修改源数据，不重新生成 IQ，不改变 20 类调制的波形。
"""

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ==================== 参数（按需修改） ====================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = (
    PROJECT_DIR / "signal_dataset_bbox_pilot100"
)

METADATA_PATH = DATASET_DIR / "metadata.json"

EXPORT_DIR = (
    PROJECT_DIR / "bbox_yolo_pilot100"
)
PREVIEW_COUNT = 5
MAX_SAMPLES = None  # None 表示全部；可设置为 5、100 等
EXPECTED_IMAGE_SIZE = (600, 400)
REQUIRE_MEASURED_BBOX = True
SPLITS = {"train", "val", "test"}

# 固定编号：后续所有数据集、训练和推理必须使用同一顺序。
CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK",
    "16QAM", "32QAM", "64QAM", "256QAM",
    "OOK", "ASK4", "ASK8",
    "2FSK", "4FSK", "8FSK", "16FSK", "GMSK",
    "OFDM_64", "OFDM_256", "FM", "AM_DSB", "AM_SSB",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def read_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"找不到 metadata: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if len(CLASS_NAMES) != 20 or len(CLASS_TO_ID) != 20:
        raise ValueError("类别数量或名称重复，请核对 CLASS_NAMES")
    if "samples" not in metadata or "frequency_range_hz" not in metadata:
        raise ValueError("metadata 缺少 samples 或 frequency_range_hz")
    return metadata


def image_path_from_metadata(raw_path):
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_DIR / path


def extract_split(image_path):
    # 生成器源路径示例：signal_dataset_bbox_measured/train/low_snr/sample_00000_...png
    parts = Path(image_path).parts
    candidates = [part for part in parts if part in SPLITS]
    if len(candidates) != 1:
        raise ValueError(f"无法唯一确认 train/val/test: {image_path}")
    return candidates[0]


def signal_to_bbox(signal_info, metadata, width, height):
    """返回浮点像素边界（边缘坐标）：[x1, y1, x2, y2]。

    保留当前验证过的全图线性映射；图像使用 origin='lower'。
    x/y 边界定义在 [0, width] / [0, height]，而非 width-1/height-1。
    """
    t0 = float(signal_info["t_start_ratio"])
    t1 = float(signal_info["t_end_ratio"])

    if REQUIRE_MEASURED_BBOX:
        low = float(signal_info["bbox_freq_low_hz"])
        high = float(signal_info["bbox_freq_high_hz"])
    else:
        low = float(signal_info.get("bbox_freq_low_hz", signal_info["freq_low_hz"]))
        high = float(signal_info.get("bbox_freq_high_hz", signal_info["freq_high_hz"]))

    f_min, f_max = (float(v) for v in metadata["frequency_range_hz"])
    values = (t0, t1, low, high, f_min, f_max)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"坐标含 NaN 或 Inf: {signal_info.get('mod_type')}")
    if not (0 <= t0 < t1 <= 1) or not (f_min < f_max) or not (low < high):
        raise ValueError(f"无效时间或频率边界: {signal_info}")

    x1 = t0 * width
    x2 = t1 * width
    y1 = (f_max - high) / (f_max - f_min) * height
    y2 = (f_max - low) / (f_max - f_min) * height

    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"边界裁剪后无效: {signal_info.get('mod_type')}")

    return [x1, y1, x2, y2]


def bbox_to_yolo(class_id, bbox, width, height):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / (2.0 * width)
    yc = (y1 + y2) / (2.0 * height)
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        raise ValueError(f"YOLO 归一化坐标非法: {bbox}")
    return f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"


def draw_preview(source_path, boxes, destination):
    with Image.open(source_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = ["#FF5555", "#55FF55", "#55AAFF", "#FFFF55", "#FF55FF", "#55FFFF"]
    w, h = canvas.size
    for index, record in enumerate(boxes):
        x1, y1, x2, y2 = record["bbox"]
        color = colors[index % len(colors)]
        # PIL rectangle 的端点是像素索引，故仅在预览绘图时裁至 width-1/height-1。
        draw.rectangle(
            [(round(x1), round(y1)),
             (min(w - 1, round(x2)), min(h - 1, round(y2)))],
            outline=color, width=2,
        )
        draw.text((int(x1) + 2, int(y1) + 2),
                  f"{index + 1}:{record['class_name']}",
                  fill=color, font=font, stroke_width=1, stroke_fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def prepare_samples(metadata):
    """先把全部待导出的样本验证完，避免发现坏框后才悄悄漏标。"""
    samples = metadata["samples"]
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    prepared = []
    used_destinations = set()
    for sample in samples:
        pair_id = sample["pair_id"]
        noisy = image_path_from_metadata(sample["low_snr_filename"])
        clean = image_path_from_metadata(sample["clean_filename"])
        if not noisy.is_file() or not clean.is_file():
            raise FileNotFoundError(f"{pair_id}: clean/noisy 文件缺失：{clean} | {noisy}")

        split = extract_split(sample["low_snr_filename"])
        if extract_split(sample["clean_filename"]) != split:
            raise ValueError(f"{pair_id}: clean/noisy 被放进不同 split")

        with Image.open(noisy) as im:
            size = im.size
        with Image.open(clean) as im:
            if im.size != size:
                raise ValueError(f"{pair_id}: clean/noisy 图像尺寸不同")
        if EXPECTED_IMAGE_SIZE is not None and size != EXPECTED_IMAGE_SIZE:
            raise ValueError(f"{pair_id}: 图像尺寸 {size} 与预期 {EXPECTED_IMAGE_SIZE} 不同")

        width, height = size
        boxes = []
        lines = []
        for signal_info in sample["signals"]:
            cls = signal_info["mod_type"]
            if cls not in CLASS_TO_ID:
                raise ValueError(f"{pair_id}: 未知类别 {cls}")
            bbox = signal_to_bbox(signal_info, metadata, width, height)
            class_id = CLASS_TO_ID[cls]
            boxes.append({"class_id": class_id, "class_name": cls, "bbox": bbox})
            lines.append(bbox_to_yolo(class_id, bbox, width, height))

        if len(boxes) != len(sample["signals"]):
            raise ValueError(f"{pair_id}: 检测框数量与信号数量不一致")

        for kind, image_path in (("low_snr", noisy), ("clean", clean)):
            key = (kind, split, image_path.stem)
            if key in used_destinations:
                raise ValueError(f"检测到相同输出文件名: {key}")
            used_destinations.add(key)

        prepared.append({
            "pair_id": pair_id, "split": split,
            "low_snr": noisy, "clean": clean,
            "labels": "\n".join(lines) + ("\n" if lines else ""),
            "boxes": boxes, "image_size": list(size),
            "signal_count": len(sample["signals"]),
        })
    return prepared


def write_dataset_yaml(base):
    # 兼容常见使用 data.yaml 的 YOLO 训练工具；旧版 YOLOv3 可能需要自定义加载器。
    content = [
        f"path: {json.dumps(str(base.resolve()), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    content.extend(f"  {i}: {json.dumps(name)}" for i, name in enumerate(CLASS_NAMES))
    (base / "data.yaml").write_text("\n".join(content) + "\n", encoding="utf-8")


def main():
    metadata = read_metadata()
    prepared = prepare_samples(metadata)
    if EXPORT_DIR.exists():
        raise FileExistsError(
            f"输出目录已存在，避免覆盖：{EXPORT_DIR}\n"
            "请先将旧目录重命名保存，或修改 EXPORT_DIR 再运行。"
        )

    for kind in ("low_snr", "clean"):
        for split in ("train", "val", "test"):
            (EXPORT_DIR / kind / "images" / split).mkdir(parents=True, exist_ok=True)
            (EXPORT_DIR / kind / "labels" / split).mkdir(parents=True, exist_ok=True)
        write_dataset_yaml(EXPORT_DIR / kind)

    preview = EXPORT_DIR / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    report = []
    split_counts = {split: 0 for split in ("train", "val", "test")}

    for i, entry in enumerate(prepared):
        split = entry["split"]
        split_counts[split] += 1
        for kind in ("low_snr", "clean"):
            image_path = entry[kind]
            destination_image = EXPORT_DIR / kind / "images" / split / image_path.name
            destination_label = EXPORT_DIR / kind / "labels" / split / (image_path.stem + ".txt")
            shutil.copy2(image_path, destination_image)
            destination_label.write_text(entry["labels"], encoding="utf-8")
            if i < PREVIEW_COUNT:
                draw_preview(image_path, entry["boxes"],
                             preview / f"{entry['pair_id']}_{kind}_bbox.png")

        report.append({
            "pair_id": entry["pair_id"], "split": split,
            "signal_count": entry["signal_count"],
            "box_count": len(entry["boxes"]),
            "image_size": entry["image_size"],
            "boxes": entry["boxes"],
        })

    (EXPORT_DIR / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )
    (EXPORT_DIR / "bbox_export_report.json").write_text(
        json.dumps({"samples": len(report), "split_counts": split_counts,
                    "class_names": CLASS_NAMES, "records": report},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"导出完成：{len(report)} 组 clean/low-SNR 配对数据")
    print("各 split 数量：", split_counts)
    print("导出目录：", EXPORT_DIR)
    print("标签行数与信号数逐一核对通过；两个域的标签文本完全相同。")


if __name__ == "__main__":
    main()
