import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBSETS_ROOT = PROJECT_ROOT / 'data_subsets'
RESULTS_ROOT = PROJECT_ROOT / 'code' / 'results'

NUM_SUBSETS = 6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, required=True, choices=range(1, NUM_SUBSETS + 1))
    args = parser.parse_args()
    subset_num = args.subset
    
    print(f"Training Subset {subset_num}/{NUM_SUBSETS}")
    
    # Load or create model
    if subset_num == 1:
        model = YOLO('yolov8n.pt')
    else:
        prev_model = RESULTS_ROOT / f'subset_{subset_num - 1}' / f'model_v{subset_num - 1}.pt'
        if not prev_model.exists():
            raise FileNotFoundError(f"Model not found: {prev_model}")
        model = YOLO(str(prev_model))
    
    # Get dataset
    dataset_yaml = SUBSETS_ROOT / f'subset_{subset_num}' / 'dataset.yaml'
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_yaml}")
    
    # Setup results
    results_dir = RESULTS_ROOT / f'subset_{subset_num}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print(f"Training...")
    model.train(
        data=str(dataset_yaml),
        epochs=100,
        batch=8,
        imgsz=416,
        device='cpu',
        optimizer='SGD',
        lr0=0.001,
        patience=10,
        project=str(results_dir.parent),
        name=f'subset_{subset_num}',
        exist_ok=True,
        verbose=True,
    )
    
    # Save model
    model_path = results_dir / f'model_v{subset_num}.pt'
    model.save(str(model_path))
    print(f"Model saved: {model_path}")
    
    # Validate
    print(f"Validating...")
    val_results = model.val(data=str(dataset_yaml), imgsz=416, device='cpu')
    
    print(f"mAP50: {val_results.box.map50:.3f} | mAP50-95: {val_results.box.map:.3f} | Precision: {val_results.box.mp:.3f} | Recall: {val_results.box.mr:.3f}")
    
    metrics_file = results_dir / f'validation_metrics_v{subset_num}.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Subset {subset_num} Validation\n")
        f.write(f"mAP50: {val_results.box.map50:.3f}\n")
        f.write(f"mAP50-95: {val_results.box.map:.3f}\n")
        f.write(f"Precision: {val_results.box.mp:.3f}\n")
        f.write(f"Recall: {val_results.box.mr:.3f}\n")
    
    if subset_num < NUM_SUBSETS:
        print(f"\nNext: python train.py --subset {subset_num + 1}")
    else:
        print(f"\nAll subsets done! Next: Evaluation")

if __name__ == '__main__':
    main()