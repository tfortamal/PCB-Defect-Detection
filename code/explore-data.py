"""
PCB Defects Dataset Exploration Script
========================================
Simple table-based analysis of PCB defect dataset
Shows: class distribution, defect sizes, defects per image
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


class PCBDataExplorer:
    # EXECUTION POINT 1: Object created here, __init__ is called
    def __init__(self, dataset_root):
        """Initialize explorer with dataset root directory"""
        self.dataset_root = Path(dataset_root)
        self.splits = ['train', 'valid', 'test']
        self.all_results = {}
    
    # EXECUTION POINT 4: Called from analyze_split() for each split
    def load_coco_annotation(self, split):
        """Load COCO JSON annotation file for a specific split"""
        annotation_path = self.dataset_root / split / '_annotations.coco.json'
        
        if not annotation_path.exists():
            print("Error: Could not find annotation file at: " + str(annotation_path))
            return None
            
        with open(annotation_path, 'r') as f:
            return json.load(f)
    
    # EXECUTION POINT 3: Called from run_full_analysis() for each split
    def analyze_split(self, split):
        """Analyze a single train/valid/test split"""
        
        # EXECUTION POINT 4: First thing inside analyze_split
        coco = self.load_coco_annotation(split)
        if not coco:
            return None
        
        # Extract data
        images = coco.get('images', [])
        annotations = coco.get('annotations', [])
        categories = {}
        for cat in coco.get('categories', []):
            categories[cat['id']] = cat['name']
        
        # Count classes
        class_counts = defaultdict(int)
        for ann in annotations:
            class_id = ann['category_id']
            class_name = categories[class_id]
            class_counts[class_name] += 1
        
        # Analyze bounding boxes
        bbox_areas = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            bbox_areas.append(w * h)
        
        # Categorize defect sizes
        tiny = 0
        small = 0
        medium = 0
        large = 0
        for a in bbox_areas:
            if a < 100:
                tiny += 1
            elif a < 500:
                small += 1
            elif a < 2000:
                medium += 1
            else:
                large += 1
        
        # Defects per image
        ann_per_img = defaultdict(int)
        for ann in annotations:
            ann_per_img[ann['image_id']] += 1
        ann_counts = list(ann_per_img.values())
        
        return {
            'split': split,
            'num_images': len(images),
            'num_annotations': len(annotations),
            'class_distribution': dict(class_counts),
            'bbox_areas': bbox_areas,
            'tiny': tiny,
            'small': small,
            'medium': medium,
            'large': large,
            'ann_per_image_mean': np.mean(ann_counts),
            'ann_per_image_max': np.max(ann_counts),
            'ann_per_image_min': np.min(ann_counts)
        }
    
    # EXECUTION POINT 6: Called from run_full_analysis() to print results
    def print_basic_numbers_table(self, split_name, result):
        """Print basic numbers in table format"""
        print("\n" + split_name.upper() + " - BASIC NUMBERS")
        print("-" * 60)
        print("Total images:        " + str(result['num_images']))
        print("Total defects found: " + str(result['num_annotations']))
        print("Number of classes:   " + str(len(result['class_distribution'])))
        print()
    
    # EXECUTION POINT 7: Called from run_full_analysis() to print results
    def print_class_distribution_table(self, result):
        """Print class distribution as a simple table"""
        print("CLASS DISTRIBUTION TABLE")
        print("-" * 60)
        print("Defect Type             Count      Percentage")
        print("-" * 60)
        
        total = result['num_annotations']
        class_dist = result['class_distribution']
        
        # Sort by count descending
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            percentage = (count / total) * 100
            print(class_name.ljust(23) + str(count).rjust(7) + str("(" + str(round(percentage, 1)) + "%)").rjust(15))
        
        print("-" * 60)
        print("TOTAL".ljust(23) + str(total).rjust(7))
        print()
    
    # EXECUTION POINT 8: Called from run_full_analysis() to print results
    def print_defect_size_table(self, result):
        """Print defect size analysis as table"""
        print("DEFECT SIZE ANALYSIS")
        print("-" * 60)
        
        bbox_areas = result['bbox_areas']
        
        print("Average defect size:    " + str(int(np.mean(bbox_areas))) + " square pixels")
        print("Median defect size:     " + str(int(np.median(bbox_areas))) + " square pixels")
        print("Min defect size:        " + str(int(np.min(bbox_areas))) + " square pixels")
        print("Max defect size:        " + str(int(np.max(bbox_areas))) + " square pixels")
        print()
        
        print("DEFECT SIZE CATEGORIES")
        print("-" * 60)
        print("Size Range              Count      Percentage")
        print("-" * 60)
        
        total = len(bbox_areas)
        tiny = result['tiny']
        small = result['small']
        medium = result['medium']
        large = result['large']
        
        print("Tiny (<100 px)".ljust(23) + str(tiny).rjust(7) + str("(" + str(int(tiny/total*100)) + "%)").rjust(15))
        print("Small (100-500 px)".ljust(23) + str(small).rjust(7) + str("(" + str(int(small/total*100)) + "%)").rjust(15))
        print("Medium (500-2000 px)".ljust(23) + str(medium).rjust(7) + str("(" + str(int(medium/total*100)) + "%)").rjust(15))
        print("Large (>2000 px)".ljust(23) + str(large).rjust(7) + str("(" + str(int(large/total*100)) + "%)").rjust(15))
        print("-" * 60)
        print("TOTAL".ljust(23) + str(total).rjust(7))
        print()
    
    # EXECUTION POINT 9: Called from run_full_analysis() to print results
    def print_defects_per_image_table(self, result):
        """Print defects per image statistics"""
        print("DEFECTS PER IMAGE")
        print("-" * 60)
        print("Average defects per image: " + str(round(result['ann_per_image_mean'], 2)))
        print("Max defects in one image:  " + str(result['ann_per_image_max']))
        print("Min defects in one image:  " + str(result['ann_per_image_min']))
        print()
    
    # EXECUTION POINT 2: Called right after object creation
    def run_full_analysis(self):
        """Analyze all splits and print results"""
        print("\n" + "="*60)
        print("PCB DEFECTS DATASET - ANALYSIS")
        print("="*60)
        
        # EXECUTION POINT 3: Loop through each split (train, valid, test)
        # Analyze each split
        results = {}
        for split in self.splits:
            result = self.analyze_split(split)  # EXECUTION POINT 3
            if result:
                results[split] = result
        
        # EXECUTION POINT 5: After all splits analyzed, print results
        # Print analysis for each split
        for split in self.splits:
            if split in results:
                result = results[split]
                self.print_basic_numbers_table(split, result)  # EXECUTION POINT 6
                self.print_class_distribution_table(result)  # EXECUTION POINT 7
                self.print_defect_size_table(result)  # EXECUTION POINT 8
                self.print_defects_per_image_table(result)  # EXECUTION POINT 9
                print()
        
        # EXECUTION POINT 10: After all splits printed, print overall summary
        # Overall summary
        print("="*60)
        print("OVERALL DATASET SUMMARY")
        print("="*60)
        
        total_images = 0
        total_annotations = 0
        for r in results.values():
            total_images += r['num_images']
            total_annotations += r['num_annotations']
        
        print("\nSplit Breakdown")
        print("-" * 60)
        print("Split               Images     Percentage")
        print("-" * 60)
        
        for split in self.splits:
            if split in results:
                result = results[split]
                pct = (result['num_images'] / total_images) * 100
                print(split.upper().ljust(19) + str(result['num_images']).rjust(7) + str("(" + str(int(pct)) + "%)").rjust(15))
        
        print("-" * 60)
        print("TOTAL".ljust(19) + str(total_images).rjust(7))
        print()
        print("Total defects in dataset: " + str(total_annotations))
        print()


# EXECUTION POINT 0: Script starts here
if __name__ == "__main__":
    # Change this to your dataset path
    DATASET_ROOT = "/Users/tamal/Desktop/PCB-Defect-Detection/data/PCB_defect.v1i.coco"
    
    explorer = PCBDataExplorer(DATASET_ROOT)  # EXECUTION POINT 1
    explorer.run_full_analysis()  # EXECUTION POINT 2