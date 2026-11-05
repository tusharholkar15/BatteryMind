"""
Script: fetch_dataset.py
Description: Retrieves the authoritative Severson et al. (MIT-Stanford-TRI) battery dataset
and corresponding experimental protocol spaces from verified scientific sources.
"""

import os
import sys
import json
import urllib.request

def fetch_authoritative_metadata():
    print("--- 1. Fetching authoritative metadata from data.matr.io ---")
    matr_files = {
        'batch1': {
            'file_id': '5c86c0b5fa2ede00015ddf66',
            'filename': '2017-05-12_batchdata_updated_struct_errorcorrect.mat'
        },
        'batch2': {
            'file_id': '5c86bf13fa2ede00015ddd82',
            'filename': '2017-06-30_batchdata_updated_struct_errorcorrect.mat'
        },
        'batch3': {
            'file_id': '5c86bd64fa2ede00015ddbb2',
            'filename': '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
        }
    }
    
    matr_meta = {}
    for k, v in matr_files.items():
        fid = v['file_id']
        url = f"https://data.matr.io/1/api/v1/file/{fid}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as resp:
                meta = json.loads(resp.read().decode('utf-8'))
                matr_meta[k] = {
                    'file_id': fid,
                    'filename': meta['name'],
                    'size_bytes': meta['size'],
                    'size_mb': round(meta['size'] / (1024 * 1024), 2),
                    'created': meta['created'],
                    'download_url': f"https://data.matr.io/1/api/v1/file/{fid}/download",
                    'project_url': "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
                }
                print(f"  Recorded {meta['name']} ({round(meta['size'] / (1024 * 1024), 2)} MB)")
        except Exception as e:
            print(f"  Warning querying {fid}: {e}")
            matr_meta[k] = {
                'file_id': fid,
                'filename': v['filename'],
                'download_url': f"https://data.matr.io/1/api/v1/file/{fid}/download",
                'project_url': "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
            }
            
    out_path = os.path.join('data', 'raw', 'matr_dataset_metadata.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(matr_meta, f, indent=2)
    print(f"  -> Saved {out_path}\n")


def fetch_cycle_lives():
    print("--- 2. Fetching cycle-life ground truth files ---")
    dest_dir = os.path.join('data', 'raw', 'cycle_lives')
    os.makedirs(dest_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/petermattia/revisit-severson-et-al/main/data/cycle_lives"
    splits = {
        'train_cycle_lives.csv': 41,
        'test1_cycle_lives.csv': 43,
        'test2_cycle_lives.csv': 40
    }
    
    for fname, expected_count in splits.items():
        url = f"{base_url}/{fname}"
        target = os.path.join(dest_dir, fname)
        urllib.request.urlretrieve(url, target)
        with open(target, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"  Downloaded {fname} ({len(lines)} cells, expected {expected_count})")
    print(f"  -> Total cells covered across splits: {sum(splits.values())}\n")


def fetch_experimental_conditions():
    print("--- 3. Fetching candidate experimental conditions (policies) ---")
    dest_path = os.path.join('data', 'raw', 'policies_all.csv')
    url = "https://raw.githubusercontent.com/chueh-ermon/battery-fast-charging-optimization/master/data/policies_all.csv"
    urllib.request.urlretrieve(url, dest_path)
    with open(dest_path, 'r', encoding='utf-8') as f:
        rows = [l.strip() for l in f if l.strip()]
    print(f"  Downloaded policies_all.csv ({len(rows)} candidate experimental protocols)")
    print(f"  -> Saved {dest_path}\n")


def fetch_cell_data():
    print("--- 4. Fetching cell-level experimental curve series ---")
    base_url = "https://raw.githubusercontent.com/petermattia/revisit-severson-et-al/main/data"
    splits = {
        'train': 41,
        'test1': 43,
        'test2': 40
    }
    
    for split, count in splits.items():
        split_dir = os.path.join('data', 'raw', split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"  Downloading {split} ({count} cells)...", end="", flush=True)
        for i in range(1, count + 1):
            cell_name = f"cell{i}.csv"
            dest = os.path.join(split_dir, cell_name)
            if not os.path.exists(dest):
                url = f"{base_url}/{split}/{cell_name}"
                urllib.request.urlretrieve(url, dest)
        print(" Done.")
    print("  -> All 124 cells downloaded successfully.\n")


if __name__ == '__main__':
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    fetch_authoritative_metadata()
    fetch_cycle_lives()
    fetch_experimental_conditions()
    fetch_cell_data()
    print("Dataset acquisition completed successfully!")
