#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit YOLO boxes with either pixel dimension below a threshold.

Matches validate_pilot100.py's tiny-box rule:
    box_height_px < 6 or box_width_px < 6
Only reads the YOLO export. Writes a separate CSV and annotated images.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Review small YOLO boxes")
    parser.add_argument("--export", type=Path, default=Path("bbox_yolo_pilot100"))
    parser.add_argument("--output", type=Path, default=Path("qa_pilot100/tiny_review"))
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--zoom", type=int, default=3)
    return parser.parse_args()


def bounds_from_yolo(xc, yc, bw, bh, width, height):
    return (
        (xc - bw / 2) * width,
        (yc - bh / 2) * height,
        (xc + bw / 2) * width,
        (yc + bh / 2) * height,
    )


def draw_full_preview(source, items, target):
    with Image.open(source) as im:
        rgb = im.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    font = ImageFont.load_default()
    width, height = rgb.size
    for item in items:
        box = tuple(item["bounds"])
        draw.rectangle(box, outline="red", width=1)
        x1, y1, _, _ = box
        label = f"#{item['label_row']} {item['class_name']}"
        draw.text((max(0, min(width-80, int(x1))), max(0, int(y1)-11)),
                  label, fill="white", font=font, stroke_width=1, stroke_fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(target)


def draw_zoom(source, item, target, zoom):
    with Image.open(source) as im:
        rgb = im.convert("RGB")
    width, height = rgb.size
    x1, y1, x2, y2 = item["bounds"]
    margin_x, margin_y = 16, 18
    left = max(0, int(x1) - margin_x)
    upper = max(0, int(y1) - margin_y)
    right = min(width, int(x2 + 0.9999) + margin_x)
    lower = min(height, int(y2 + 0.9999) + margin_y)
    if right <= left or lower <= upper:
        raise ValueError("Invalid crop geometry")
    crop = rgb.crop((left, upper, right, lower))
    resampling = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    enlarged = crop.resize((crop.width * zoom, crop.height * zoom), resample=resampling)
    draw = ImageDraw.Draw(enlarged)
    draw.rectangle(((x1-left)*zoom, (y1-upper)*zoom,
                    (x2-left)*zoom, (y2-upper)*zoom), outline="red", width=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    enlarged.save(target)


def main():
    args = parse_args()
    export = args.export
    output = args.output
    classes_path = export / "classes.txt"
    if not classes_path.is_file():
        raise SystemExit(f"Missing class definitions: {classes_path}")
    names = classes_path.read_text(encoding="utf-8").splitlines()
    if len(names) != 20:
        raise SystemExit(f"Expected 20 classes, found {len(names)}")

    records = []
    per_image = defaultdict(list)
    for split in ("train", "val", "test"):
        label_dir = export / "clean" / "labels" / split
        image_dir = export / "clean" / "images" / split
        if not label_dir.is_dir():
            raise SystemExit(f"Missing label directory: {label_dir}")
        for label_file in sorted(label_dir.glob("*.txt")):
            image_file = image_dir / f"{label_file.stem}.png"
            if not image_file.is_file():
                raise SystemExit(f"Missing image for {label_file}: {image_file}")
            with Image.open(image_file) as im:
                width, height = im.size
            for row_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
                tokens = line.split()
                if len(tokens) != 5:
                    raise SystemExit(f"Invalid 5-column YOLO row: {label_file}:{row_number}")
                cls = int(tokens[0])
                if not 0 <= cls < len(names):
                    raise SystemExit(f"Invalid class ID: {label_file}:{row_number}")
                xc, yc, bw, bh = map(float, tokens[1:])
                width_px, height_px = bw * width, bh * height
                if not (width_px < args.threshold or height_px < args.threshold):
                    continue
                item = {
                    "split": split,
                    "pair_id": label_file.stem.replace("_clean", ""),
                    "label_row": row_number,
                    "class_id": cls,
                    "class_name": names[cls],
                    "width_px": width_px,
                    "height_px": height_px,
                    "bounds": bounds_from_yolo(xc, yc, bw, bh, width, height),
                    "image": image_file,
                    "label": label_file,
                }
                records.append(item)
                per_image[image_file].append(item)

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "tiny_boxes.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "pair_id", "label_row", "class_id", "class_name",
                         "width_px", "height_px", "clean_image", "review(PASS/FAIL)", "notes"])
        for item in records:
            writer.writerow([item["split"], item["pair_id"], item["label_row"],
                             item["class_id"], item["class_name"],
                             f"{item['width_px']:.2f}", f"{item['height_px']:.2f}",
                             str(item["image"]), "", ""])

    for image_path, items in per_image.items():
        split = items[0]["split"]
        draw_full_preview(image_path, items,
                          output / "full" / f"{split}_{image_path.stem}_tiny.png")
        for item in items:
            dst = output / "zoom" / (
                f"{split}_{item['pair_id']}_row{item['label_row']:02d}_"
                f"{item['class_name']}.png")
            draw_zoom(image_path, item, dst, args.zoom)

    print("Tiny-box criterion: width_px <", args.threshold,
          "OR height_px <", args.threshold)
    print("Tiny boxes:", len(records))
    print("Affected clean images:", len(per_image))
    print("Class breakdown:", dict(sorted(Counter(r["class_name"] for r in records).items())))
    print("Review CSV:", csv_path)
    print("Full previews:", output / "full")
    print("Zoom previews:", output / "zoom")
    if len(records) != 35:
        print("NOTE: Previous QA found 35; verify export path and that labels have not changed.")


if __name__ == "__main__":
    main()
