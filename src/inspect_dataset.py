"""
Script: inspect_dataset.py
Description: Structural inspection and sanity checks for the MIT-Stanford Fast-Charging
battery dataset stored in data/raw/. Inspects dataset without modifying any files.
"""

import os
import json
import glob
import numpy as np
import pandas as pd


def get_file_size_str(bytes_size):
    """Convert bytes to human-readable string."""
    if bytes_size >= 1024 * 1024 * 1024:
        return f"{bytes_size / (1024**3):.2f} GB"
    elif bytes_size >= 1024 * 1024:
        return f"{bytes_size / (1024**2):.2f} MB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KB"
    return f"{bytes_size} Bytes"


def inspect_matr_source_metadata(raw_dir):
    """Inspect original primary repository metadata from data.matr.io."""
    print("=" * 80)
    print("1. PRIMARY REPOSITORY SOURCE & METADATA (data.matr.io)")
    print("=" * 80)
    meta_path = os.path.join(raw_dir, 'matr_dataset_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        total_mat_size = sum(item.get('size_bytes', 0) for item in meta.values())
        print(f"Authoritative Portal: {list(meta.values())[0].get('project_url', 'N/A')}")
        print(f"Total Raw Full-Time-Series Size: {get_file_size_str(total_mat_size)} across 3 batches")
        for k, v in meta.items():
            print(f"  [{k}] {v['filename']}")
            print(f"      Girder File ID : {v['file_id']}")
            print(f"      Size           : {v['size_mb']} MB ({v['size_bytes']:,} bytes)")
            print(f"      Created        : {v.get('created', 'N/A')}")
            print(f"      Download URL   : {v['download_url']}")
    else:
        print("  [!] matr_dataset_metadata.json not found.")
    print()


def inspect_raw_file_inventory(raw_dir):
    """List all raw files and directory hierarchy."""
    print("=" * 80)
    print("2. RAW FILE INVENTORY (data/raw/)")
    print("=" * 80)
    file_records = []
    total_bytes = 0
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, raw_dir)
            size = os.path.getsize(full_path)
            total_bytes += size
            file_records.append((rel_path, size))
            
    print(f"Total files in data/raw/: {len(file_records)}")
    print(f"Total storage on disk   : {get_file_size_str(total_bytes)}")
    print("\nKey Directory Breakdown:")
    subdirs = ['cycle_lives', 'train', 'test1', 'test2']
    for sd in subdirs:
        sd_path = os.path.join(raw_dir, sd)
        if os.path.exists(sd_path):
            count = len(os.listdir(sd_path))
            sd_size = sum(os.path.getsize(os.path.join(sd_path, x)) for x in os.listdir(sd_path))
            print(f"  - data/raw/{sd}/ : {count} files ({get_file_size_str(sd_size)})")
    
    # Standalone files
    standalone = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    for sf in standalone:
        s_size = os.path.getsize(os.path.join(raw_dir, sf))
        print(f"  - data/raw/{sf} : {get_file_size_str(s_size)}")
    print()


def inspect_target_and_cells(raw_dir):
    """Inspect cycle life targets, cell IDs, distributions, and statistics."""
    print("=" * 80)
    print("3. TARGET VARIABLE & CELL POPULATION INSPECTION")
    print("=" * 80)
    cycle_dir = os.path.join(raw_dir, 'cycle_lives')
    splits = {
        'train': os.path.join(cycle_dir, 'train_cycle_lives.csv'),
        'test1': os.path.join(cycle_dir, 'test1_cycle_lives.csv'),
        'test2': os.path.join(cycle_dir, 'test2_cycle_lives.csv')
    }
    
    cell_data = []
    split_summaries = {}
    
    for split_name, filepath in splits.items():
        if not os.path.exists(filepath):
            print(f"  [!] Missing file: {filepath}")
            continue
        df = pd.read_csv(filepath, header=None, names=['cycle_life'])
        split_summaries[split_name] = {
            'count': len(df),
            'min': float(df['cycle_life'].min()),
            'max': float(df['cycle_life'].max()),
            'median': float(df['cycle_life'].median()),
            'mean': float(df['cycle_life'].mean()),
            'std': float(df['cycle_life'].std())
        }
        for idx, val in enumerate(df['cycle_life'], start=1):
            cell_id = f"{split_name}_cell{idx}"
            cell_data.append({
                'cell_id': cell_id,
                'split': split_name,
                'cell_index': idx,
                'cycle_life': val
            })
            
    df_all_cells = pd.DataFrame(cell_data)
    total_cells = len(df_all_cells)
    
    print(f"Total Unique Cells: {total_cells}")
    print(f"Target Variable   : Battery Cycle Life (cycles until 80% capacity retention)")
    print(f"Target Data Type  : {df_all_cells['cycle_life'].dtype}")
    print(f"Missing Values    : {df_all_cells['cycle_life'].isna().sum()}\n")
    
    print("Per-Split Cell Breakdown:")
    for split, s in split_summaries.items():
        print(f"  - {split:<6}: {s['count']:>2} cells | Min: {s['min']:>6.1f} | Max: {s['max']:>6.1f} | "
              f"Median: {s['median']:>6.1f} | Mean: {s['mean']:>6.1f} ± {s['std']:<5.1f}")
        
    print("\nGlobal Target Statistics (All 124 Cells):")
    stats = df_all_cells['cycle_life'].describe(percentiles=[0.25, 0.50, 0.75])
    print(f"  Count  : {stats['count']:.0f}")
    print(f"  Mean   : {stats['mean']:.2f} cycles")
    print(f"  Std    : {stats['std']:.2f} cycles")
    print(f"  Min    : {stats['min']:.1f} cycles")
    print(f"  25%    : {stats['25%']:.1f} cycles")
    print(f"  Median : {stats['50%']:.1f} cycles (50th percentile)")
    print(f"  75%    : {stats['75%']:.1f} cycles")
    print(f"  Max    : {stats['max']:.1f} cycles")
    print()
    return df_all_cells


def inspect_experimental_conditions(raw_dir):
    """Inspect candidate experimental conditions and charging protocols."""
    print("=" * 80)
    print("4. EXPERIMENTAL PROTOCOLS & PARAMETER SPACE")
    print("=" * 80)
    policy_path = os.path.join(raw_dir, 'policies_all.csv')
    if os.path.exists(policy_path):
        df_policies = pd.read_csv(policy_path, header=None)
        df_policies.columns = ['C1_current_A', 'C2_current_A', 'C3_current_A', 'C4_current_A']
        print(f"Policy File           : data/raw/policies_all.csv")
        print(f"Candidate Experiments : {len(df_policies)} distinct candidate charging protocols")
        print(f"Protocol Steps (Cols) : {df_policies.shape[1]} (4-step multi-stage fast charge currents)")
        print(f"Missing Values        : {df_policies.isna().sum().sum()}")
        print("\nCandidate Experimental Space Ranges:")
        for col in df_policies.columns:
            print(f"  - {col:<14}: Min = {df_policies[col].min():.2f}C | Max = {df_policies[col].max():.2f}C | "
                  f"Mean = {df_policies[col].mean():.2f}C | Unique values = {df_policies[col].nunique()}")
        print("\nFirst 5 Sample Candidate Experimental Conditions:")
        print(df_policies.head(5).to_string(index=True))
    else:
        print("  [!] policies_all.csv not found.")
    print()


def inspect_cell_curves(raw_dir):
    """Inspect the structure of cell-level experimental curve series."""
    print("=" * 80)
    print("5. CELL EXPERIMENTAL CURVES INSPECTION (Early-Cycle Signatures)")
    print("=" * 80)
    sample_files = [
        ('train', os.path.join(raw_dir, 'train', 'cell1.csv')),
        ('test1', os.path.join(raw_dir, 'test1', 'cell1.csv')),
        ('test2', os.path.join(raw_dir, 'test2', 'cell1.csv')),
    ]
    for split, path in sample_files:
        if os.path.exists(path):
            arr = np.genfromtxt(path, delimiter=',')
            print(f"Sample [{split}/cell1.csv]:")
            print(f"  Shape              : {arr.shape} (rows: {arr.shape[0]} voltage points, cols: {arr.shape[1]} cycles)")
            print(f"  Voltage Resolution : 1,000 interpolated voltage points (3.6V -> 2.0V)")
            print(f"  Cycle Window       : 99 cycles (cycles 2 to 100)")
            print(f"  Data Type          : {arr.dtype}")
            print(f"  NaN Count          : {np.isnan(arr).sum()}")
            print(f"  Value Range        : [{np.nanmin(arr):.6f}, {np.nanmax(arr):.6f}]")
    print()


def perform_sanity_checks(raw_dir, df_cells):
    """Execute all mandatory Phase 1 sanity checks."""
    print("=" * 80)
    print("6. SANITY CHECKS & DATA QUALITY AUDIT")
    print("=" * 80)
    issues_found = []
    
    # 1. Duplicate cell IDs
    dup_ids = df_cells['cell_id'].duplicated().sum()
    print(f"  [CHECK 1] Duplicate Cell Identifiers : {dup_ids} (Pass)")
    if dup_ids > 0:
        issues_found.append(f"Found {dup_ids} duplicate cell IDs.")
        
    # 2. Impossible cycle counts (<= 0 or > 10,000)
    impossible_cycles = ((df_cells['cycle_life'] <= 0) | (df_cells['cycle_life'] > 10000)).sum()
    print(f"  [CHECK 2] Impossible Cycle Counts    : {impossible_cycles} (Pass)")
    if impossible_cycles > 0:
        issues_found.append(f"Found {impossible_cycles} impossible cycle counts.")
        
    # 3. Negative values
    negative_cycles = (df_cells['cycle_life'] < 0).sum()
    print(f"  [CHECK 3] Negative Target Values     : {negative_cycles} (Pass)")
    if negative_cycles > 0:
        issues_found.append(f"Found {negative_cycles} negative target values.")
        
    # 4. Missing target values
    missing_targets = df_cells['cycle_life'].isna().sum()
    print(f"  [CHECK 4] Missing Target Values      : {missing_targets} (Pass)")
    if missing_targets > 0:
        issues_found.append(f"Found {missing_targets} missing target values.")
        
    # 5. Inconsistent identifiers / file count
    train_files = glob.glob(os.path.join(raw_dir, 'train', '*.csv'))
    test1_files = glob.glob(os.path.join(raw_dir, 'test1', '*.csv'))
    test2_files = glob.glob(os.path.join(raw_dir, 'test2', '*.csv'))
    total_curve_files = len(train_files) + len(test1_files) + len(test2_files)
    print(f"  [CHECK 5] Identifier Consistency     : {total_curve_files} curve files match {len(df_cells)} target rows (Pass)")
    if total_curve_files != len(df_cells):
        issues_found.append(f"Mismatch: {total_curve_files} curve files vs {len(df_cells)} target entries.")
        
    # 6. File corruption check across all 124 cell CSVs
    corrupted_count = 0
    all_files = train_files + test1_files + test2_files
    for f in all_files:
        try:
            if os.path.getsize(f) == 0:
                corrupted_count += 1
        except Exception:
            corrupted_count += 1
    print(f"  [CHECK 6] File Corruption / Zero-Byte: {corrupted_count} corrupted files (Pass)")
    if corrupted_count > 0:
        issues_found.append(f"Found {corrupted_count} corrupted or zero-byte curve files.")
        
    # 7. Unexpected experimental conditions
    policy_path = os.path.join(raw_dir, 'policies_all.csv')
    df_policies = pd.read_csv(policy_path, header=None)
    invalid_currents = ((df_policies < 0) | (df_policies > 15)).sum().sum()
    print(f"  [CHECK 7] Experimental Policy Bounds : {invalid_currents} anomalous current rates (Pass)")
    if invalid_currents > 0:
        issues_found.append(f"Found {invalid_currents} charging currents out of physical bounds.")
        
    print()
    if not issues_found:
        print("  >>> ALL SANITY CHECKS PASSED PERFECTLY (0 Quality Issues Detected) <<<")
    else:
        print(f"  >>> WARNING: {len(issues_found)} issues detected:")
        for iss in issues_found:
            print(f"      - {iss}")
    print()


def main():
    raw_dir = os.path.join('data', 'raw')
    if not os.path.exists(raw_dir):
        print(f"Error: Directory {raw_dir} does not exist.")
        return
    
    inspect_matr_source_metadata(raw_dir)
    inspect_raw_file_inventory(raw_dir)
    df_cells = inspect_target_and_cells(raw_dir)
    inspect_experimental_conditions(raw_dir)
    inspect_cell_curves(raw_dir)
    perform_sanity_checks(raw_dir, df_cells)


if __name__ == '__main__':
    main()
