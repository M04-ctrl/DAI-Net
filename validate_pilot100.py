#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QA for DAI-Net-Signal 100-pair YOLO pilot (read-only inputs).

Usage:
  python validate_pilot100.py \
    --dataset signal_dataset_bbox_pilot100 \
    --export bbox_yolo_pilot100 \
    --expected 100

Reads source metadata and YOLO export; writes summary and a small set of
clean/noisy annotated previews into a separate --report directory.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAMES = [
    'BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM', '256QAM',
    'OOK', 'ASK4', 'ASK8', '2FSK', '4FSK', '8FSK', '16FSK', 'GMSK',
    'OFDM_64', 'OFDM_256', 'FM', 'AM_DSB', 'AM_SSB',
]
SPLITS = ('train', 'val', 'test')
DOMAIN_KEYS = (('low_snr', 'low_snr_filename'), ('clean', 'clean_filename'))
IMAGE_SIZE = (600, 400)
TOL = 2e-6


def clip(x):
    return min(1.0, max(0.0, x))


def split_from_path(raw):
    matches = [x for x in Path(raw).parts if x in SPLITS]
    if len(matches) != 1:
        raise ValueError('cannot uniquely determine split from ' + str(raw))
    return matches[0]


def expected_yolo(info, f_min, f_max):
    t0, t1 = float(info['t_start_ratio']), float(info['t_end_ratio'])
    fl = float(info['bbox_freq_low_hz'])
    fh = float(info['bbox_freq_high_hz'])
    if not all(math.isfinite(v) for v in (t0, t1, fl, fh)):
        raise ValueError('non-finite metadata bbox')
    if not (0 <= t0 < t1 <= 1 and fl < fh):
        raise ValueError('invalid metadata bbox geometry')
    x1, x2 = clip(t0), clip(t1)
    y1, y2 = clip((f_max-fh)/(f_max-f_min)), clip((f_max-fl)/(f_max-f_min))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('zero-size metadata bbox after clipping')
    return ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1), (x1, y1, x2, y2)


def read_image_size(path):
    with Image.open(path) as image:
        image.load()
        return image.size


def preview(src, parsed, dst):
    with Image.open(src) as im:
        im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    w, h = im.size
    colors = ('#fa5151', '#63f36e', '#4cc7ff', '#fff157', '#f26aff', '#55ffee')
    for i, (cls, xc, yc, bw, bh) in enumerate(parsed):
        x1, x2 = (xc-bw/2)*w, (xc+bw/2)*w
        y1, y2 = (yc-bh/2)*h, (yc+bh/2)*h
        color = colors[i % len(colors)]
        draw.rectangle((round(x1), round(y1), min(w-1, round(x2)),
                        min(h-1, round(y2))), outline=color, width=2)
        draw.text((min(w-1, round(x1)+2), max(0, round(y1)+2)),
                  str(i+1)+':'+NAMES[cls], fill=color, font=font,
                  stroke_width=1, stroke_fill='black')
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=Path('signal_dataset_bbox_pilot100'))
    ap.add_argument('--export', type=Path, default=Path('bbox_yolo_pilot100'))
    ap.add_argument('--report', type=Path, default=Path('qa_pilot100'))
    ap.add_argument('--expected', type=int, default=100)
    ap.add_argument('--preview-count', type=int, default=12)
    args = ap.parse_args()

    errors, warnings = [], []
    metadata_file = args.dataset / 'metadata.json'
    if not metadata_file.exists():
        raise SystemExit('missing metadata: ' + str(metadata_file))
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    samples = metadata.get('samples', [])
    if len(samples) != args.expected:
        errors.append(f'metadata has {len(samples)} pairs; expected {args.expected}')
    try:
        f_min, f_max = [float(v) for v in metadata['frequency_range_hz']]
        if not f_min < f_max:
            raise ValueError('invalid frequency range')
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit('invalid metadata frequency_range_hz: ' + str(exc))

    names_file = args.export / 'classes.txt'
    if not names_file.is_file():
        errors.append('missing ' + str(names_file))
    elif names_file.read_text(encoding='utf-8').splitlines() != NAMES:
        errors.append('classes.txt does not match fixed 20-class mapping')

    for domain, _ in DOMAIN_KEYS:
        config = args.export / domain / 'data.yaml'
        if not config.is_file():
            errors.append('missing ' + str(config))
        else:
            data = config.read_text(encoding='utf-8')
            for term in ('nc: 20', 'train: images/train', 'val: images/val',
                         'test: images/test'):
                if term not in data:
                    errors.append(f'{config} missing {term}')

    class_counts = Counter()
    snr_counts = Counter()
    split_counts = Counter()
    total_boxes = 0
    seen_pairs = set()
    expected_stems = defaultdict(set)
    previews = []
    risk_counts = Counter()

    for sample in samples:
        pid = str(sample.get('pair_id', 'unknown'))
        if pid in seen_pairs:
            errors.append('duplicate pair_id: ' + pid)
            continue
        seen_pairs.add(pid)
        try:
            paths = {d: Path(sample[key]) for d, key in DOMAIN_KEYS}
            split = split_from_path(paths['low_snr'])
            if split_from_path(paths['clean']) != split:
                raise ValueError('clean/noisy cross-split')
            split_counts[split] += 1
            snr_counts[str(sample.get('low_snr_db', 'unknown'))] += 1
            expected_count = len(sample['signals'])
            if not (5 <= expected_count <= 20):
                warnings.append(f'{pid}: {expected_count} signals (expected 5..20)')

            parsed_by_domain = {}
            label_bytes = {}
            label_paths = {}
            for domain, _ in DOMAIN_KEYS:
                image = args.export / domain / 'images' / split / paths[domain].name
                label = args.export / domain / 'labels' / split / (paths[domain].stem + '.txt')
                expected_stems[(domain, split)].add(paths[domain].stem)
                if not image.is_file() or not label.is_file():
                    raise FileNotFoundError(f'missing image/label: {image} | {label}')
                size = read_image_size(image)
                if size != IMAGE_SIZE:
                    errors.append(f'{pid}: {domain} image size {size} (expected {IMAGE_SIZE})')
                lines = label.read_text(encoding='utf-8').splitlines()
                if len(lines) != expected_count:
                    errors.append(f'{pid}: {domain} labels={len(lines)} but signals={expected_count}')
                parsed = []
                for i, line in enumerate(lines):
                    tokens = line.split()
                    if len(tokens) != 5:
                        errors.append(f'{pid}: {domain} row {i} has {len(tokens)} columns')
                        continue
                    try:
                        raw_id = float(tokens[0])
                        vals = tuple(float(x) for x in tokens[1:])
                        if not math.isfinite(raw_id) or not raw_id.is_integer() or not (0 <= raw_id < 20):
                            raise ValueError('class ID invalid')
                        if not all(math.isfinite(v) for v in vals):
                            raise ValueError('non-finite coords')
                        xc, yc, bw, bh = vals
                        if not (0 < bw <= 1 and 0 < bh <= 1
                                and 0 <= xc-bw/2+1e-7 and xc+bw/2 <= 1+1e-7
                                and 0 <= yc-bh/2+1e-7 and yc+bh/2 <= 1+1e-7):
                            raise ValueError('box out of range or zero size')
                        parsed.append((int(raw_id), *vals))
                        if i < expected_count:
                            meta_info = sample['signals'][i]
                            name = meta_info['mod_type']
                            if name not in NAMES or NAMES[int(raw_id)] != name:
                                errors.append(f'{pid}: row {i} class mismatches metadata: {name}')
                            expect_vals, _ = expected_yolo(meta_info, f_min, f_max)
                            if max(abs(a-b) for a, b in zip(expect_vals, vals)) > TOL:
                                errors.append(f'{pid}: row {i} coordinates differ from metadata')
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        errors.append(f'{pid}: {domain} row {i}: {exc}')
                parsed_by_domain[domain] = parsed
                label_bytes[domain] = label.read_bytes()
                label_paths[domain] = label
            if label_bytes.get('clean') != label_bytes.get('low_snr'):
                errors.append(f'{pid}: clean/noisy labels differ')

            # Count each object once, since both domains share labels.
            for cls, xc, yc, bw, bh in parsed_by_domain.get('low_snr', []):
                class_counts[NAMES[cls]] += 1
                total_boxes += 1
                if bh*IMAGE_SIZE[1] < 6 or bw*IMAGE_SIZE[0] < 6:
                    risk_counts['tiny_boxes_under_6px'] += 1
                    warnings.append(f'{pid}: tiny {NAMES[cls]} box (<6 px)')
                if min(xc-bw/2, yc-bh/2, 1-xc-bw/2, 1-yc-bh/2) < 0.002:
                    risk_counts['near_image_edge'] += 1
            parsed = parsed_by_domain.get('low_snr', [])
            previews.append((sample, split, parsed))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors.append(f'{pid}: {exc}')

    for domain, _ in DOMAIN_KEYS:
        for split in SPLITS:
            image_dir = args.export / domain / 'images' / split
            label_dir = args.export / domain / 'labels' / split
            actual_images = {p.stem for p in image_dir.glob('*.png')} if image_dir.exists() else set()
            actual_labels = {p.stem for p in label_dir.glob('*.txt')} if label_dir.exists() else set()
            expected = expected_stems[(domain, split)]
            if actual_images != expected:
                errors.append(f'{domain}/{split}: image stems do not match metadata '
                              f'(missing {len(expected-actual_images)}, extra {len(actual_images-expected)})')
            if actual_labels != expected:
                errors.append(f'{domain}/{split}: label stems do not match metadata '
                              f'(missing {len(expected-actual_labels)}, extra {len(actual_labels-expected)})')

    if len(class_counts) < 20:
        warnings.append(f'only {len(class_counts)}/20 classes appear in this pilot')
    if sum(split_counts.values()) != args.expected:
        errors.append(f'split count total {sum(split_counts.values())}, expected {args.expected}')
    if any(split_counts[s] == 0 for s in SPLITS):
        warnings.append('one split is empty (not suitable for full train/val/test smoke test)')

    # Choose examples for human inspection: class coverage + dense and sparse scenes.
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / 'preview').mkdir(exist_ok=True)
    valid = [p for p in previews if p[2]]
    selected = []
    covered = set()
    pool = list(valid)
    while pool and len(selected) < args.preview_count:
        best = max(pool, key=lambda rec: (
            4*len({box[0] for box in rec[2]}-covered),
            0.1*len(rec[2]),
            -len(rec[2]) if len(selected) % 3 == 1 else len(rec[2])
        ))
        selected.append(best)
        covered.update(b[0] for b in best[2])
        pool.remove(best)
    for sample, split, parsed in selected:
        for domain, key in DOMAIN_KEYS:
            image = args.export / domain / 'images' / split / Path(sample[key]).name
            if image.exists():
                out = args.report / 'preview' / f"{sample['pair_id']}_{domain}.png"
                preview(image, parsed, out)

    with (args.report / 'class_distribution.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'object_count'])
        for idx, name in enumerate(NAMES):
            writer.writerow([idx, name, class_counts[name]])

    report = {
        'dataset': str(args.dataset), 'export': str(args.export),
        'expected_pairs': args.expected, 'metadata_pairs': len(samples),
        'split_pairs': {k: split_counts[k] for k in SPLITS},
        'snr_pairs': dict(sorted(snr_counts.items())),
        'total_objects': total_boxes,
        'class_distribution': {name: class_counts[name] for name in NAMES},
        'risk_counts': dict(risk_counts),
        'errors': errors, 'warnings': warnings,
        'preview_pairs': [v[0]['pair_id'] for v in selected],
    }
    (args.report / 'validation_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Source pairs:', len(samples))
    print('Split pairs:', {s: split_counts[s] for s in SPLITS})
    print('SNR counts:', dict(sorted(snr_counts.items())))
    print('Total labeled objects (one domain):', total_boxes)
    print('Classes represented:', len(class_counts), '/ 20')
    print('Potential risk counts:', dict(risk_counts))
    print('Errors:', len(errors), '| Warnings:', len(warnings))
    for e in errors[:15]:
        print('ERROR:', e)
    for w in warnings[:12]:
        print('WARN:', w)
    print('QA output:', args.report)
    if errors:
        raise SystemExit(1)
    print('PASS: structural and metadata label checks')


if __name__ == '__main__':
    main()
