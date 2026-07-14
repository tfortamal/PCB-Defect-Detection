import json
from pathlib import Path
from collections import defaultdict
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / 'data' / 'PCB_defect.v1i.coco'
OUTPUT_ROOT = PROJECT_ROOT / 'data_subsets'

NUM_SUBSETS = 6
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

def load_coco(split):
    path = DATASET_ROOT / split / '_annotations.coco.json'
    print(f"Loading {split}...")
    with open(path) as f:
        return json.load(f)

def get_img_anns(coco):
    result = defaultdict(list)
    for ann in coco['annotations']:
        result[ann['image_id']].append(ann)
    return dict(result)

def group_by_class(coco, img_anns):
    result = defaultdict(set)
    for img_id, anns in img_anns.items():
        for ann in anns:
            result[ann['category_id']].add(img_id)
    return {k: list(v) for k, v in result.items()}

def split_images(classes, num_subsets):
    subsets = [set() for _ in range(num_subsets)]
    for class_id in sorted(classes.keys()):
        imgs = classes[class_id]
        random.shuffle(imgs)
        for i, img_id in enumerate(imgs):
            subsets[i % num_subsets].add(img_id)
    return subsets

def to_yolo_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    return [(x + bw/2) / w, (y + bh/2) / h, bw / w, bh / h]

def make_dirs(subset_num):
    d = OUTPUT_ROOT / f'subset_{subset_num}'
    (d / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (d / 'images' / 'valid').mkdir(parents=True, exist_ok=True)
    (d / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (d / 'labels' / 'valid').mkdir(parents=True, exist_ok=True)
    return d

def copy_and_label(subset_dir, split, img_ids, coco_data, img_anns, src_split):
    id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    count = 0
    
    for img_id in img_ids:
        if img_id not in id_to_file:
            continue
        
        count += 1
        fname = id_to_file[img_id]
        src = DATASET_ROOT / src_split / fname
        dst = subset_dir / 'images' / split / fname
        
        if not dst.exists():
            shutil.copy2(src, dst)
        
        img_info = next((i for i in coco_data['images'] if i['id'] == img_id), None)
        if not img_info:
            continue
        
        anns = img_anns.get(img_id, [])
        label_file = subset_dir / 'labels' / split / fname.replace('.jpg', '.txt')
        
        with open(label_file, 'w') as f:
            for ann in anns:
                cls = ann['category_id'] - 1
                bbox = to_yolo_bbox(ann['bbox'], img_info['width'], img_info['height'])
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")
    
    return count

def write_yaml(subset_num, subset_dir):
    yaml = f"""path: {subset_dir}
train: images/train
val: images/valid
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(subset_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml)

def main():
    print("Splitting dataset...")
    
    train = load_coco('train')
    valid = load_coco('valid')
    
    train_anns = get_img_anns(train)
    valid_anns = get_img_anns(valid)
    
    train_cls = group_by_class(train, train_anns)
    valid_cls = group_by_class(valid, valid_anns)
    
    train_sub = split_images(train_cls, NUM_SUBSETS)
    valid_sub = split_images(valid_cls, NUM_SUBSETS)
    
    combined = [train_sub[i] | valid_sub[i] for i in range(NUM_SUBSETS)]
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    for n in range(1, NUM_SUBSETS + 1):
        print(f"\nSubset {n}:")
        d = make_dirs(n)
        
        t = copy_and_label(d, 'train', train_sub[n-1], train, train_anns, 'train')
        v = copy_and_label(d, 'valid', valid_sub[n-1], valid, valid_anns, 'valid')
        
        print(f"  Train: {t}, Valid: {v}")
        write_yaml(n, d)
    
    print("\nDone")

if __name__ == '__main__':
    main()
