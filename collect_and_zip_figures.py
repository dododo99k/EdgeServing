"""
Collect all PNG files from figures_* directories and create a ZIP archive.

This script:
1. Finds all directories matching 'figures_*'
2. Collects PNG files from the first level of each directory
3. Copies them to a new folder 'all_figures'
4. Creates a ZIP archive 'all_figures.zip'
"""

import os
import glob
import shutil
import zipfile
from pathlib import Path


def collect_and_zip_figures(output_folder="all_figures", zip_name="all_figures.zip"):
    """
    Collect PNG files from all figures_* directories and create a ZIP archive.
    
    Parameters
    ----------
    output_folder : str
        Folder to collect all PNG files
    zip_name : str
        Name of the output ZIP file
    """
    # Create output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all figures_* directories
    figures_dirs = glob.glob("figures_*")
    figures_dirs = [d for d in figures_dirs if os.path.isdir(d)]
    
    if not figures_dirs:
        print("No figures_* directories found!")
        return
    
    print(f"Found {len(figures_dirs)} figures_* directories:")
    for d in sorted(figures_dirs):
        print(f"  - {d}")
    
    # Collect PNG files
    total_files = 0
    for fig_dir in sorted(figures_dirs):
        # Get PNG files in first level only
        png_files = glob.glob(os.path.join(fig_dir, "*.png"))
        
        if not png_files:
            print(f"\nNo PNG files in {fig_dir}")
            continue
        
        print(f"\nProcessing {fig_dir}: found {len(png_files)} PNG files")
        
        for png_file in png_files:
            # Keep original file name
            base_name = os.path.basename(png_file)
            dest_path = os.path.join(output_folder, base_name)
            
            # Copy file
            shutil.copy2(png_file, dest_path)
            print(f"  Copied: {base_name}")
            total_files += 1
    
    print(f"\n{'='*60}")
    print(f"Total files collected: {total_files}")
    print(f"Output folder: {output_folder}")
    
    # Create ZIP archive
    if total_files > 0:
        print(f"\nCreating ZIP archive: {zip_name}")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)
                    print(f"  Added to ZIP: {arcname}")
        
        zip_size = os.path.getsize(zip_name) / (1024 * 1024)  # MB
        print(f"\n{'='*60}")
        print(f"ZIP archive created: {zip_name}")
        print(f"Archive size: {zip_size:.2f} MB")
        print(f"{'='*60}")
    else:
        print("\nNo files to archive!")


if __name__ == "__main__":
    collect_and_zip_figures()
