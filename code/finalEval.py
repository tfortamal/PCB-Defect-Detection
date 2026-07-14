from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / 'data' / 'yolo_format'
TEST_DIR = DATA_ROOT / 'images' / 'test'
TEST_LABELS = DATA_ROOT / 'labels' / 'test'
RESULTS_ROOT = PROJECT_ROOT / 'code' / 'results'

# Model to evaluate
model_path = RESULTS_ROOT / 'subset_6' / 'weights' / 'best.pt'

def main():
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Evaluating on test set: {TEST_DIR}")
    print(f"Test labels: {TEST_LABELS}")
    print(f"Test images: 2,127 (completely unseen)\n")
    
    # Create test dataset.yaml
    test_yaml = DATA_ROOT / 'dataset_test.yaml'
    if not test_yaml.exists():
        yaml_content = """path: {0}
train: images/test
val: images/test
nc: 6
names: ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
""".format(DATA_ROOT)
        with open(test_yaml, 'w') as f:
            f.write(yaml_content)
    
    # Run evaluation
    results = model.val(data=str(test_yaml), imgsz=416, device='cpu')
    
    # Print results
    print()
    print("="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:   {results.box.mr:.4f}")
    print("="*60)
    
    # Per-class results
    print("\nPer-Class Performance:")
    print("-"*60)
    classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    for i, class_name in enumerate(classes):
        precision = results.box.mp_per_class[i]
        recall = results.box.mr_per_class[i]
        ap50 = results.box.map50_per_class[i]
        print(f"{class_name:20s} | Precision: {precision:.4f} | Recall: {recall:.4f} | AP50: {ap50:.4f}")
    print("-"*60)
    
    # Save results
    results_file = RESULTS_ROOT / 'final_test_results' / 'test_metrics.txt'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("FINAL TEST SET EVALUATION\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Images: 2,127 (unseen)\n\n")
        f.write(f"mAP50:    {results.box.map50:.4f}\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
        f.write(f"Precision: {results.box.mp:.4f}\n")
        f.write(f"Recall:   {results.box.mr:.4f}\n\n")
        f.write("Per-Class Results:\n")
        f.write("-"*60 + "\n")
        for i, class_name in enumerate(classes):
            precision = results.box.mp_per_class[i]
            recall = results.box.mr_per_class[i]
            ap50 = results.box.map50_per_class[i]
            f.write(f"{class_name:20s} | Precision: {precision:.4f} | Recall: {recall:.4f} | AP50: {ap50:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()