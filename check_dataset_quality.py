"""
DATASET QUALITY CHECKER
Run this to check if your dataset actually has enough tumors
This will tell you if the problem is the data itself
"""

import nibabel as nib
import numpy as np
import os

print("="*70)
print("🔬 DATASET QUALITY ANALYSIS")
print("="*70)
print("Analyzing your lung tumor dataset to find the root cause...\n")

# =============================================================================
# ANALYZE RAW DATA (before any preprocessing)
# =============================================================================

IMAGES_DIR = "Task06_Lung/imagesTr"
LABELS_DIR = "Task06_Lung/labelsTr"

# Get all files
label_files = sorted([f for f in os.listdir(LABELS_DIR) 
                      if f.endswith('.nii.gz') and not f.startswith('._')])

print(f"Found {len(label_files)} label files\n")
print("Analyzing tumor content in each file...")
print("-"*70)

tumor_stats = []
no_tumor_files = []
small_tumor_files = []

for i, label_file in enumerate(label_files[:10]):  # Check first 10
    label_path = os.path.join(LABELS_DIR, label_file)
    
    try:
        # Load label
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata()
        
        # Calculate tumor statistics
        total_voxels = label_data.size
        tumor_voxels = (label_data > 0).sum()
        tumor_pct = (tumor_voxels / total_voxels) * 100
        
        tumor_stats.append({
            'file': label_file,
            'tumor_voxels': int(tumor_voxels),
            'total_voxels': int(total_voxels),
            'tumor_pct': tumor_pct
        })
        
        # Categorize
        if tumor_voxels == 0:
            no_tumor_files.append(label_file)
            status = "❌ NO TUMOR"
        elif tumor_pct < 0.1:
            small_tumor_files.append(label_file)
            status = "⚠️  VERY SMALL"
        elif tumor_pct < 1.0:
            status = "⚡ Small"
        else:
            status = "✅ Good"
        
        print(f"{i+1:2d}. {label_file}")
        print(f"    Tumor: {tumor_voxels:,} / {total_voxels:,} voxels ({tumor_pct:.3f}%) {status}")
        
    except Exception as e:
        print(f"{i+1:2d}. {label_file} - ERROR: {str(e)}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("📊 DATASET SUMMARY")
print("="*70)

if tumor_stats:
    tumor_percentages = [s['tumor_pct'] for s in tumor_stats]
    tumor_voxel_counts = [s['tumor_voxels'] for s in tumor_stats]
    
    print(f"\nFiles analyzed: {len(tumor_stats)}")
    print(f"Files with NO tumor: {len(no_tumor_files)}")
    print(f"Files with VERY SMALL tumors (<0.1%): {len(small_tumor_files)}")
    print()
    print(f"Tumor percentage statistics:")
    print(f"  Average: {np.mean(tumor_percentages):.3f}%")
    print(f"  Median:  {np.median(tumor_percentages):.3f}%")
    print(f"  Min:     {np.min(tumor_percentages):.3f}%")
    print(f"  Max:     {np.max(tumor_percentages):.3f}%")
    print()
    print(f"Tumor voxel count:")
    print(f"  Average: {np.mean(tumor_voxel_counts):,.0f}")
    print(f"  Median:  {np.median(tumor_voxel_counts):,.0f}")
    print(f"  Min:     {np.min(tumor_voxel_counts):,.0f}")
    print(f"  Max:     {np.max(tumor_voxel_counts):,.0f}")

# =============================================================================
# DIAGNOSIS
# =============================================================================
print("\n" + "="*70)
print("🔍 DIAGNOSIS")
print("="*70)

if len(no_tumor_files) > len(tumor_stats) * 0.5:
    print("\n❌ CRITICAL ISSUE:")
    print(f"   {len(no_tumor_files)}/{len(tumor_stats)} files have NO tumor!")
    print("   Your label files may be corrupted or empty.")
    print()
    print("🔧 SOLUTION:")
    print("   Re-download the dataset from AWS:")
    print("   wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar")
    
elif np.mean(tumor_percentages) < 0.5:
    print("\n⚠️  DATASET HAS VERY SMALL TUMORS:")
    print(f"   Average tumor percentage: {np.mean(tumor_percentages):.3f}%")
    print("   This is why training is so difficult!")
    print()
    print("🔧 SOLUTIONS:")
    print("   1. Use VERY aggressive cropping (pos=19, neg=1 = 95% tumor)")
    print("   2. Use smaller patches (32x32x32 or 48x48x48)")
    print("   3. Use extremely weighted loss (focal weight=50)")
    print("   4. Train for longer (100+ epochs)")
    print()
    print("   Expected Dice with this dataset: 0.3-0.5 (not great but realistic)")
    
elif np.mean(tumor_percentages) < 2.0:
    print("\n⚡ DATASET HAS SMALL TUMORS (but workable):")
    print(f"   Average tumor percentage: {np.mean(tumor_percentages):.3f}%")
    print()
    print("🔧 SOLUTIONS:")
    print("   1. Use aggressive cropping (pos=9, neg=1)")
    print("   2. Use smaller patches (48x48x48)")
    print("   3. Heavily weighted loss (focal weight=25)")
    print()
    print("   Expected Dice: 0.4-0.6")
    
else:
    print("\n✅ DATASET LOOKS GOOD:")
    print(f"   Average tumor percentage: {np.mean(tumor_percentages):.3f}%")
    print()
    print("   If you're still getting low Dice scores, the problem is likely:")
    print("   1. Preprocessing not using tumor-focused cropping")
    print("   2. Loss function not aggressive enough")
    print("   3. Model architecture issues")
    print()
    print("   Expected Dice: 0.5-0.7")

# =============================================================================
# SPECIFIC RECOMMENDATIONS
# =============================================================================
print("\n" + "="*70)
print("💡 SPECIFIC RECOMMENDATIONS FOR YOUR DATASET")
print("="*70)

avg_tumor_pct = np.mean(tumor_percentages)

if avg_tumor_pct < 0.5:
    print("\nRECOMMENDED SETTINGS:")
    print("  SPATIAL_SIZE = (32, 32, 32)  # Very small patches")
    print("  POS_NEG_RATIO = 19  # 95% tumor patches")
    print("  Focal weight = 50")
    print("  Epochs = 100+")
    
elif avg_tumor_pct < 2.0:
    print("\nRECOMMENDED SETTINGS:")
    print("  SPATIAL_SIZE = (48, 48, 48)  # Small patches")
    print("  POS_NEG_RATIO = 9   # 90% tumor patches")
    print("  Focal weight = 25")
    print("  Epochs = 50-75")
    
else:
    print("\nRECOMMENDED SETTINGS:")
    print("  SPATIAL_SIZE = (64, 64, 64)  # Normal patches")
    print("  POS_NEG_RATIO = 5   # 83% tumor patches")
    print("  Focal weight = 15")
    print("  Epochs = 40-50")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
