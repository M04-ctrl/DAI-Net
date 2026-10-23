
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAI-Net-Signal: Bounding Box 标注预览 v0.1

功能：
1. 读取原有 metadata.json
2. 根据时间和频率参数生成候选 Bounding Box
3. 在 clean / low SNR 图片上绘制检测框
4. 保存前5组样本的预览图和候选坐标

注意：
当前使用近似全图线性映射，需人工验证。
不修改原始数据集，不生成正式训练标签。
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ========== 1. 路径设置 ==========

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR / "signal_dataset_bbox_measured"
)

METADATA_PATH = DATASET_DIR / "metadata.json"

OUTPUT_DIR = (
    PROJECT_DIR / "bbox_preview_measured"
)

NUM_PREVIEW = 5


# ========== 2. 读取元数据 ==========

def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ========== 3. 生成候选 Bounding Box ==========

def signal_to_bbox(signal_info, metadata, width, height):
    """
    将信号时间、频率参数转换为图像坐标。

    返回：
        [xmin, ymin, xmax, ymax]

    坐标原点：图像左上角
    x 向右增大，y 向下增大

    说明：
    目前假定整个PNG画布对应完整的时间与频率范围。
    实际Matplotlib图像映射仍需可视化核验。
    """

    # 读取时间比例
    t_start = signal_info["t_start_ratio"]
    t_end = signal_info["t_end_ratio"]

    # 读取频率参数
    f_low = signal_info.get(
    "bbox_freq_low_hz",
    signal_info["freq_low_hz"]
    )

    f_high = signal_info.get(
        "bbox_freq_high_hz",
        signal_info["freq_high_hz"]
    )

    # 整体采样频率范围
    f_min, f_max = metadata["frequency_range_hz"]
    freq_span = f_max - f_min

    # 时间 -> 水平像素坐标
    x1 = t_start * width
    x2 = t_end * width

    # 频率 -> 垂直像素坐标
    # origin='lower'：高频位于上方
    y1 = (f_max - f_high) / freq_span * height
    y2 = (f_max - f_low) / freq_span * height

    # 限制坐标范围
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    bbox = [
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2))
    ]

    return bbox


# ========== 4. 在谱图上绘制检测框 ==========

def draw_bboxes(image_path, boxes, output_path):

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    colors = [
        "#FF5555",
        "#55FF55",
        "#55AAFF",
        "#FFFF55",
        "#FF55FF",
        "#55FFFF",
    ]

    for index, item in enumerate(boxes):

        bbox = item["bbox"]
        label = item["class_name"]

        color = colors[index % len(colors)]

        x1, y1, x2, y2 = bbox

        # 绘制目标框
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2
        )

        # 显示目标序号和调制类别
        text = f"{index + 1}:{label}"

        # 将文字放在框内顶部，避免超出画布
        text_x = x1 + 2
        text_y = y1 + 2

        draw.text(
            (text_x, text_y),
            text,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="black"
        )

    image.save(output_path)


# ========== 5. 处理单个样本 ==========

def process_sample(sample, metadata):

    pair_id = sample["pair_id"]
    snr = sample["low_snr_db"]

    # generate.py保存的相对路径以项目目录为起点
    noisy_path = PROJECT_DIR / sample["low_snr_filename"]
    clean_path = PROJECT_DIR / sample["clean_filename"]

    if not noisy_path.exists():
        print("缺失低SNR图片:", noisy_path)
        return None

    if not clean_path.exists():
        print("缺失纯净图片:", clean_path)
        return None

    with Image.open(noisy_path) as image:
        width, height = image.size

    with Image.open(clean_path) as image:
        assert image.size == (width, height), (
            f"{pair_id}: 成对图片尺寸不一致"
        )

    boxes = []

    # 每个信号分别生成一个候选目标框
    for signal_info in sample["signals"]:

        bbox = signal_to_bbox(
            signal_info,
            metadata,
            width,
            height
        )

        # 跳过无效框
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            print("警告：无效框", pair_id, bbox)
            continue

        boxes.append({
            "class_name": signal_info["mod_type"],
            "bbox": bbox
        })

    # 保存预览图片
    clean_output = OUTPUT_DIR / f"{pair_id}_clean_bbox.png"
    noisy_output = OUTPUT_DIR / f"{pair_id}_SNR{snr}dB_bbox.png"

    draw_bboxes(clean_path, boxes, clean_output)
    draw_bboxes(noisy_path, boxes, noisy_output)

    print(
        f"{pair_id}: SNR={snr}dB, "
        f"信号数={len(sample['signals'])}, "
        f"候选框数={len(boxes)}"
    )

    return {
        "pair_id": pair_id,
        "snr_db": snr,
        "image_size": [width, height],
        "boxes": boxes
    }


# ========== 6. 主程序 ==========

def main():

    print("=" * 60)
    print("DAI-Net-Signal: Bounding Box 预览")
    print("=" * 60)

    metadata = load_metadata()
    samples = metadata["samples"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("总样本数:", len(samples))
    print("本次预览数量:", NUM_PREVIEW)
    print("输出目录:", OUTPUT_DIR)

    results = []

    for sample in samples[:NUM_PREVIEW]:

        result = process_sample(sample, metadata)

        if result is not None:
            results.append(result)

    # 保存候选框的JSON，便于后续检查
    output_json = OUTPUT_DIR / "preview_boxes.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 60)
    print("预览生成完成！")
    print("成功处理:", len(results), "组")
    print("预览保存于:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
