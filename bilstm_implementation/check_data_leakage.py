"""
Data Leakage Detection Script

This script checks for potential data leakage issues in the BiLSTM implementation.
"""

import os
import re
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_numeric_id(filename):
    """Extract the integer timestamp embedded in the filename"""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def analyze_temporal_distribution():
    """Analyze the temporal distribution of different classes"""
    
    data_dir = Path("d:/MTECH/data/Adani_MTF_Images_224x224")
    classes = ['BUY', 'HOLD', 'SELL']
    
    class_data = {}
    all_entries = []
    
    print("Analyzing temporal distribution of classes...")
    print("=" * 60)
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
            
        ids = []
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                numeric_id = extract_numeric_id(fname)
                ids.append(numeric_id)
                all_entries.append((numeric_id, class_name))
        
        ids = sorted(ids)
        class_data[class_name] = ids
        
        print(f"{class_name}:")
        print(f"  Count: {len(ids)}")
        print(f"  Range: {min(ids)} - {max(ids)}")
        print(f"  First 10: {ids[:10]}")
        print(f"  Last 10: {ids[-10:]}")
        print()
    
    # Sort all entries by timestamp
    all_entries.sort(key=lambda x: x[0])
    
    print("Temporal Analysis:")
    print("=" * 60)
    
    # Check for temporal clustering
    temporal_segments = defaultdict(list)
    
    # Divide timeline into segments
    all_timestamps = [entry[0] for entry in all_entries]
    min_time, max_time = min(all_timestamps), max(all_timestamps)
    
    # Create 10 temporal segments
    segment_size = (max_time - min_time) // 10
    
    for timestamp, class_name in all_entries:
        segment = (timestamp - min_time) // segment_size
        if segment >= 10:  # Handle edge case
            segment = 9
        temporal_segments[segment].append(class_name)
    
    print("Class distribution across temporal segments:")
    print("Segment | BUY   | HOLD  | SELL  | Total")
    print("--------|-------|-------|-------|-------")
    
    segment_stats = []
    for i in range(10):
        classes_in_segment = temporal_segments[i]
        buy_count = classes_in_segment.count('BUY')
        hold_count = classes_in_segment.count('HOLD')
        sell_count = classes_in_segment.count('SELL')
        total = len(classes_in_segment)
        
        print(f"   {i:2d}   | {buy_count:5d} | {hold_count:5d} | {sell_count:5d} | {total:5d}")
        
        if total > 0:
            segment_stats.append({
                'segment': i,
                'buy_pct': buy_count / total * 100,
                'hold_pct': hold_count / total * 100,
                'sell_pct': sell_count / total * 100
            })
    
    print("\nPercentage distribution per segment:")
    print("Segment | BUY%   | HOLD%  | SELL%")
    print("--------|--------|--------|--------")
    
    for stats in segment_stats:
        print(f"   {stats['segment']:2d}   | {stats['buy_pct']:6.1f} | {stats['hold_pct']:6.1f} | {stats['sell_pct']:6.1f}")
    
    # Check for potential leakage issues
    print("\nPotential Leakage Issues:")
    print("=" * 60)
    
    # Check 1: Are classes heavily clustered in time?
    max_class_concentration = 0
    problematic_segments = []
    
    for stats in segment_stats:
        max_pct = max(stats['buy_pct'], stats['hold_pct'], stats['sell_pct'])
        if max_pct > 80:  # More than 80% of one class in a segment
            problematic_segments.append((stats['segment'], max_pct))
        max_class_concentration = max(max_class_concentration, max_pct)
    
    print(f"Maximum class concentration in any segment: {max_class_concentration:.1f}%")
    
    if problematic_segments:
        print("⚠️  WARNING: Heavily biased temporal segments found:")
        for segment, pct in problematic_segments:
            print(f"   Segment {segment}: {pct:.1f}% dominated by one class")
    else:
        print("✓ No heavily biased temporal segments found")
    
    # Check 2: Sequence overlap analysis
    print(f"\nSequence Overlap Analysis (sequence_length=10):")
    print("=" * 60)
    
    sequence_length = 10
    total_possible_sequences = len(all_entries) - sequence_length + 1
    
    print(f"Total images: {len(all_entries)}")
    print(f"Total possible sequences: {total_possible_sequences}")
    
    # Simulate temporal split (64/16/20)
    train_end = int(len(all_entries) * 0.64)
    val_end = int(len(all_entries) * 0.80)
    
    print(f"Train sequences: 0 to {train_end - sequence_length + 1}")
    print(f"Val sequences: {train_end - sequence_length + 1} to {val_end - sequence_length + 1}")
    print(f"Test sequences: {val_end - sequence_length + 1} to {total_possible_sequences}")
    
    # Check for overlap
    train_images_end = train_end
    val_images_start = train_end
    val_images_end = val_end  
    test_images_start = val_end
    
    print(f"\nImage index ranges:")
    print(f"Train images: 0 to {train_images_end-1}")
    print(f"Val images: {val_images_start} to {val_images_end-1}")
    print(f"Test images: {test_images_start} to {len(all_entries)-1}")
    
    # Calculate overlap
    train_seq_last_img = train_end - 1 + (sequence_length - 1)
    if train_seq_last_img >= val_images_start:
        overlap = train_seq_last_img - val_images_start + 1
        print(f"⚠️  WARNING: Train sequences use images {val_images_start} to {train_seq_last_img}")
        print(f"   This overlaps with validation by {overlap} images!")
    
    val_seq_last_img = val_end - 1 + (sequence_length - 1) 
    if val_seq_last_img >= test_images_start:
        overlap = val_seq_last_img - test_images_start + 1
        print(f"⚠️  WARNING: Val sequences use images {test_images_start} to {val_seq_last_img}")
        print(f"   This overlaps with test set by {overlap} images!")
        
    return all_entries, class_data

def check_sequence_labels():
    """Check how sequence labels are assigned and if there's look-ahead bias"""
    
    print("\nSequence Label Assignment Analysis:")
    print("=" * 60)
    
    # Simulate the sequence creation process
    data_dir = Path("d:/MTECH/data/Adani_MTF_Images_224x224")
    classes = ['BUY', 'HOLD', 'SELL']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    all_entries = []
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                numeric_id = extract_numeric_id(fname)
                all_entries.append((numeric_id, fname, class_to_idx[class_name]))
    
    # Sort by timestamp
    all_entries.sort(key=lambda x: x[0])
    
    sequence_length = 10
    
    print(f"Analyzing first 20 sequences (length={sequence_length}):")
    print("Seq# | Images in sequence                    | Label")
    print("-----|---------------------------------------|--------")
    
    for i in range(min(20, len(all_entries) - sequence_length + 1)):
        sequence_files = []
        sequence_classes = []
        
        for j in range(sequence_length):
            entry = all_entries[i + j]
            filename = entry[1]
            class_label = entry[2]
            sequence_files.append(filename)
            sequence_classes.append(classes[class_label])
        
        # Label is from the LAST image
        final_label = sequence_classes[-1]
        
        # Show class progression
        class_progression = "->".join(sequence_classes)
        if len(class_progression) > 35:
            class_progression = class_progression[:32] + "..."
        
        print(f"{i:4d} | {class_progression:<35} | {final_label}")
    
    # Analyze label distribution
    sequence_labels = []
    class_changes_in_sequences = []
    
    for i in range(len(all_entries) - sequence_length + 1):
        sequence_classes = []
        for j in range(sequence_length):
            class_label = all_entries[i + j][2]
            sequence_classes.append(class_label)
        
        # Count unique classes in sequence
        unique_classes = len(set(sequence_classes))
        class_changes_in_sequences.append(unique_classes)
        
        # Label from last image
        sequence_labels.append(sequence_classes[-1])
    
    print(f"\nSequence Diversity Analysis:")
    print(f"Total sequences: {len(sequence_labels)}")
    
    from collections import Counter
    label_dist = Counter(sequence_labels)
    class_diversity_dist = Counter(class_changes_in_sequences)
    
    print(f"\nFinal label distribution:")
    for class_idx, count in label_dist.items():
        class_name = classes[class_idx]
        pct = count / len(sequence_labels) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    print(f"\nClass diversity within sequences:")
    for num_classes, count in sorted(class_diversity_dist.items()):
        pct = count / len(class_changes_in_sequences) * 100
        if num_classes == 1:
            print(f"  {num_classes} class (homogeneous): {count} ({pct:.1f}%)")
        else:
            print(f"  {num_classes} classes (mixed): {count} ({pct:.1f}%)")
    
    # Calculate percentage of homogeneous sequences
    homogeneous_pct = class_diversity_dist[1] / len(class_changes_in_sequences) * 100
    
    if homogeneous_pct > 50:
        print(f"⚠️  WARNING: {homogeneous_pct:.1f}% of sequences are homogeneous!")
        print("   This makes the task artificially easy and unrealistic.")
    else:
        print(f"✓ Only {homogeneous_pct:.1f}% of sequences are homogeneous")

if __name__ == "__main__":
    print("BiLSTM Data Leakage Analysis")
    print("=" * 80)
    
    all_entries, class_data = analyze_temporal_distribution()
    check_sequence_labels()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nIf you see any of these warnings, your results may be unrealistic:")
    print("1. Heavy temporal clustering of classes")
    print("2. Overlapping images between train/val/test splits")
    print("3. High percentage of homogeneous sequences")
    print("4. Unrealistic accuracy (>90% for financial time series)")
    
    print("\nRecommendations to fix data leakage:")
    print("1. Ensure proper temporal splitting with no overlap")
    print("2. Use shorter sequences to reduce homogeneity")
    print("3. Consider using different data generation methods")
    print("4. Validate results against financial domain knowledge")
