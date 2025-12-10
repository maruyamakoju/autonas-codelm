#!/usr/bin/env python3
"""
Download HumanEval and MBPP datasets.

Usage:
    python nas/tools/download_datasets.py
"""

import urllib.request
import gzip
import shutil
from pathlib import Path


def download_file(url: str, output_path: Path):
    """Download file from URL."""
    print(f"Downloading {url}...")
    print(f"  → {output_path}")

    try:
        with urllib.request.urlopen(url) as response:
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response, f)
        print(f"  [OK] Downloaded ({output_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        return False


def main():
    """Download HumanEval and MBPP datasets."""
    output_dir = Path("data/external")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            'name': 'HumanEval',
            'url': 'https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz',
            'output': output_dir / 'HumanEval.jsonl.gz',
            'decompress': True
        },
        {
            'name': 'MBPP',
            'url': 'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl',
            'output': output_dir / 'mbpp.jsonl',
            'decompress': False
        }
    ]

    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset['name']}")
        print('='*70)

        output_path = dataset['output']

        # Skip if already exists
        final_path = output_path.with_suffix('') if dataset['decompress'] else output_path
        if final_path.exists():
            print(f"  [SKIP] Already exists: {final_path}")
            continue

        # Download
        success = download_file(dataset['url'], output_path)
        if not success:
            continue

        # Decompress if needed
        if dataset['decompress']:
            print(f"Decompressing...")
            decompressed_path = output_path.with_suffix('')
            try:
                with gzip.open(output_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  [OK] Decompressed -> {decompressed_path}")

                # Remove .gz file
                output_path.unlink()
                print(f"  [CLEAN] Removed {output_path.name}")
            except Exception as e:
                print(f"  [ERROR] Decompression failed: {e}")

    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print('='*70)

    # List downloaded files
    print("\nDownloaded files:")
    for f in sorted(output_dir.glob('*.jsonl')):
        size_kb = f.stat().st_size / 1024
        print(f"  [OK] {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
