"""
Script: extract_features.py
Description: Feature extraction pipeline for BatteryMind (Phase 3).
Extracts the approved, leakage-free early-cycle degradation features (up to Cycle 100)
from the raw MIT-Stanford battery dataset and creates data/processed/batterymind_features.csv.
"""

import os
import sys
import numpy as np
import pandas as pd


def extract_battery_features(raw_dir='data/raw', out_dir='data/processed'):
    """
    Extracts the 5 approved baseline features for all 124 battery cells:
      1. log10_var_deltaQ : log10 variance of ΔQ100-10(V) curve
      2. min_deltaQ       : minimum capacity difference along ΔQ100-10(V)
      3. mean_deltaQ      : mean capacity difference along ΔQ100-10(V)
      4. q_cycle100       : discharge capacity at 2.0V cutoff of Cycle 100
      5. fade_100_10      : capacity fade at 2.0V cutoff (Cycle 100 - Cycle 10)
    
    Includes cell metadata:
      - cell_id           : unique identifier (e.g. train_cell1, test1_cell1)
      - dataset_split     : designated cohort ('train', 'test1', 'test2')
      - cycle_life        : ground truth target (cycles to 80% nominal capacity)
      
    Note on charging policies:
      The 224 rows in policies_all.csv define the continuous experimental design space
      from the Attia et al. 2020 study, but individual cell-to-policy rows are not
      unambiguously tagged per cell in raw curves. To avoid unverified guesswork,
      policy columns are omitted from this core electrochemical feature table.
    """
    print("=" * 80)
    print("BATTERYMIND: FEATURE EXTRACTION PIPELINE (PHASE 3)")
    print("=" * 80)
    
    cycle_dir = os.path.join(raw_dir, 'cycle_lives')
    splits = ['train', 'test1', 'test2']
    
    records = []
    
    # Process each partition
    for split in splits:
        target_path = os.path.join(cycle_dir, f"{split}_cycle_lives.csv")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")
            
        targets = pd.read_csv(target_path, header=None)[0].values
        split_dir = os.path.join(raw_dir, split)
        
        print(f"Processing partition [{split}]: {len(targets)} cells expected...")
        
        for idx, y in enumerate(targets, start=1):
            cell_id = f"{split}_cell{idx}"
            curve_file = os.path.join(split_dir, f"cell{idx}.csv")
            
            if not os.path.exists(curve_file):
                raise FileNotFoundError(f"Missing curve file for {cell_id}: {curve_file}")
                
            # Load curve matrix: shape (1000, 99)
            # 1,000 voltage points along Vdlin (3.6V -> 2.0V)
            # 99 cycles: columns 0 to 98 correspond to Cycle 2 to Cycle 100
            mat = np.genfromtxt(curve_file, delimiter=',')
            
            if mat.shape != (1000, 99):
                raise ValueError(f"Unexpected matrix shape for {cell_id}: {mat.shape}, expected (1000, 99)")
                
            # Cycle 10 is at column index 8 (col 0 = cycle 2)
            # Cycle 100 is at column index 98
            q10 = mat[:, 8]
            q100 = mat[:, 98]
            dq = q100 - q10
            
            # 1. log10_var_deltaQ: sample variance across 1,000 voltage points
            var_dq = float(np.var(dq, ddof=1))
            if var_dq <= 0:
                raise ValueError(f"Non-positive variance for {cell_id}: {var_dq}")
            log10_var = float(np.log10(var_dq))
            
            # 2. min_deltaQ: minimum point along the ΔQ100-10 curve
            min_dq = float(np.min(dq))
            
            # 3. mean_deltaQ: mean point along the ΔQ100-10 curve
            mean_dq = float(np.mean(dq))
            
            # 4. q_cycle100: discharge capacity at 2.0V cutoff (last row = index 999) of Cycle 100
            q_c100 = float(mat[-1, 98])
            
            # 5. fade_100_10: capacity difference at 2.0V cutoff between Cycle 100 and Cycle 10
            fade_100_10 = float(mat[-1, 98] - mat[-1, 8])
            
            records.append({
                'cell_id': cell_id,
                'dataset_split': split,
                'cycle_life': int(y),
                'log10_var_deltaQ': log10_var,
                'min_deltaQ': min_dq,
                'mean_deltaQ': mean_dq,
                'q_cycle100': q_c100,
                'fade_100_10': fade_100_10
            })
            
    df_features = pd.DataFrame(records)
    
    # Validation checks
    print("\n--- Validating Extracted Features ---")
    print(f"Total Rows (Cells)       : {len(df_features)}")
    print(f"Total Columns            : {len(df_features.columns)}")
    print(f"Unique Cell IDs          : {df_features['cell_id'].nunique()}")
    print(f"Missing Values Count     : {df_features.isna().sum().sum()}")
    num_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10', 'cycle_life']
    inf_count = np.isinf(df_features[num_cols]).sum().sum()
    print(f"Infinite Values Count    : {inf_count}")
    
    if df_features['cell_id'].nunique() != len(df_features):
        raise ValueError("Duplicate cell_id detected in extracted features!")
    if df_features.isna().sum().sum() > 0:
        raise ValueError("Missing values detected in extracted features!")
    if inf_count > 0:
        raise ValueError("Infinite values detected in extracted features!")
        
    # Save to data/processed
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'batterymind_features.csv')
    df_features.to_csv(out_path, index=False)
    print(f"\n-> Successfully saved processed dataset: {out_path} ({os.path.getsize(out_path)} bytes)")
    
    return df_features


if __name__ == '__main__':
    extract_battery_features()
