"""
Convert old Pointcept .pth format to new directory-based .npy format for ScanNet.

Usage:
    python scripts/convert_pth_to_npy.py \
        --input_root /path/to/old/scannet \
        --output_root /path/to/new/scannet
"""

import os
import argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def convert_scene(args):
    pth_path, output_root, split = args
    try:
        data = torch.load(pth_path, weights_only=False, map_location='cpu')

        scene_name = os.path.basename(pth_path).replace('.pth', '')
        scene_dir = os.path.join(output_root, split, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        # coord
        np.save(os.path.join(scene_dir, 'coord.npy'), data['coord'].astype(np.float32))

        # color
        color = data['color']
        if color.dtype == np.uint8:
            color = color.astype(np.float32)
        np.save(os.path.join(scene_dir, 'color.npy'), color)

        # normal
        np.save(os.path.join(scene_dir, 'normal.npy'), data['normal'].astype(np.float32))

        # segment20
        if 'semantic_gt20' in data:
            np.save(os.path.join(scene_dir, 'segment20.npy'), data['semantic_gt20'].astype(np.int32))

        # segment200
        if 'semantic_gt200' in data:
            np.save(os.path.join(scene_dir, 'segment200.npy'), data['semantic_gt200'].astype(np.int32))

        # instance
        if 'instance_gt' in data:
            np.save(os.path.join(scene_dir, 'instance.npy'), data['instance_gt'].astype(np.int32))

        return scene_name, True, None
    except Exception as e:
        return pth_path, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.input_root, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (not found)")
            continue

        pth_files = glob(os.path.join(split_dir, '*.pth'))
        if not pth_files:
            print(f"Skipping {split} (no .pth files)")
            continue

        print(f"\nConverting {split}: {len(pth_files)} scenes")
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

        tasks = [(p, args.output_root, split) for p in pth_files]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(executor.map(convert_scene, tasks), total=len(tasks)))

        success = sum(1 for _, s, _ in results if s)
        print(f"{split}: {success}/{len(pth_files)} converted")

    print(f"\nDone! Output: {args.output_root}")


if __name__ == '__main__':
    main()
