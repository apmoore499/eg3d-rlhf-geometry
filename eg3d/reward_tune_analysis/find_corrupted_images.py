#!/usr/bin/env python3
"""
Script to identify all corrupted images in a ZIP dataset.
This will check all images and report which ones are corrupted.
"""

import os
import sys
import zipfile
import io
from pathlib import Path

import numpy as np
import PIL.Image

try:
    import pyspng
except ImportError:
    pyspng = None


def check_image(zf, fname, use_pyspng=False):
    """Check if an image file can be loaded successfully."""
    try:
        # Try to open and read the file from ZIP
        try:
            f = zf.open(fname, "r")
        except zipfile.BadZipFile as e:
            if "Overlapped entries" in str(e) or "possible zip bomb" in str(e):
                # Use read() as workaround
                data = zf.read(fname)
                f = io.BytesIO(data)
            else:
                return False, f"BadZipFile: {str(e)}"

        with f:
            # Try to load the image
            if use_pyspng and fname.lower().endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))

        # Check image is valid
        if image.size == 0:
            return False, "Empty image"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def find_corrupted_images(zip_path):
    """Find all corrupted images in a ZIP file."""
    print(f"Checking ZIP file: {zip_path}")
    print("-" * 80)

    if not os.path.exists(zip_path):
        print(f"ERROR: ZIP file not found: {zip_path}")
        return

    corrupted_files = []
    valid_files = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        image_files = [name for name in zf.namelist()
                      if os.path.splitext(name.lower())[1] in image_extensions
                      and not name.startswith('__MACOSX')]

        print(f"Found {len(image_files)} image files to check\n")

        # Check each image
        for idx, fname in enumerate(image_files):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(image_files)}", end='\r')

            is_valid, msg = check_image(zf, fname, use_pyspng=(pyspng is not None))

            if is_valid:
                valid_files.append(fname)
            else:
                corrupted_files.append((idx, fname, msg))
                print(f"\n[CORRUPTED] Index {idx}: {fname}")
                print(f"            Error: {msg}")

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images:     {len(image_files)}")
    print(f"Valid images:     {len(valid_files)}")
    print(f"Corrupted images: {len(corrupted_files)}")

    if corrupted_files:
        print("\n" + "=" * 80)
        print("CORRUPTED FILES LIST")
        print("=" * 80)
        for idx, fname, msg in corrupted_files:
            print(f"Index {idx:6d}: {fname}")
            print(f"              Error: {msg}")

        # Save to file
        output_file = zip_path.replace('.zip', '_corrupted_list.txt')
        with open(output_file, 'w') as f:
            f.write("Corrupted Images Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"ZIP File: {zip_path}\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Corrupted images: {len(corrupted_files)}\n\n")
            for idx, fname, msg in corrupted_files:
                f.write(f"Index {idx}: {fname}\n")
                f.write(f"  Error: {msg}\n\n")

        print(f"\nCorrupted files list saved to: {output_file}")
    else:
        print("\nNo corrupted files found!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_corrupted_images.py <path_to_zip_file>")
        print("\nExample:")
        print("  python find_corrupted_images.py /path/to/dataset.zip")
        sys.exit(1)

    zip_path = sys.argv[1]
    find_corrupted_images(zip_path)
