"""
COMPREHENSIVE DATA VERIFICATION SCRIPT
Run this after recreating your dataloaders to verify everything is correct
"""

import torch
import numpy as np

print("="*70)
print("🔍 COMPREHENSIVE DATA VERIFICATION")
print("="*70)

# =============================================================================
# 1. CHECK TRAINING DATA (Multiple Batches)
# =============================================================================
print("\n📊 TRAINING DATA ANALYSIS (checking 5 batches):")
print("-"*70)

train_tumor_percentages = []
train_tumor_voxels = []

for i, batch in enumerate(train_loader):
    if i >= 5:  # Check first 5 batches
        break
    
    if isinstance(batch, list):
        batch = batch[0]
    
    image = batch["image"]
    label = batch["label"]
    
    tumor_voxels = (label > 0).sum().item()
    total_voxels = label.numel()
    tumor_pct = (tumor_voxels / total_voxels) * 100
    
    train_tumor_percentages.append(tumor_pct)
    train_tumor_voxels.append(tumor_voxels)
    
    print(f"\nBatch {i+1}:")
    print(f"  Shape: {image.shape} (batch_size, channels, D, H, W)")
    print(f"  Tumor voxels: {tumor_voxels:,} / {total_voxels:,}")
    print(f"  Tumor %: {tumor_pct:.2f}%")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Check if model could learn from this
    if tumor_pct < 1.0:
        print(f"  ⚠️  WARNING: Very low tumor percentage!")
    elif tumor_pct < 5.0:
        print(f"  ⚡ Low but workable")
    else:
        print(f"  ✅ Good tumor percentage for training")

avg_train_tumor = np.mean(train_tumor_percentages)
print(f"\n📈 Training Data Summary:")
print(f"  Average tumor %: {avg_train_tumor:.2f}%")
print(f"  Min tumor %: {min(train_tumor_percentages):.2f}%")
print(f"  Max tumor %: {max(train_tumor_percentages):.2f}%")

# =============================================================================
# 2. CHECK VALIDATION DATA (Multiple Batches)
# =============================================================================
print("\n" + "="*70)
print("📊 VALIDATION DATA ANALYSIS (checking 5 batches):")
print("-"*70)

val_tumor_percentages = []
val_tumor_voxels = []

for i, batch in enumerate(val_loader):
    if i >= 5:
        break
    
    image = batch["image"]
    label = batch["label"]
    
    tumor_voxels = (label > 0).sum().item()
    total_voxels = label.numel()
    tumor_pct = (tumor_voxels / total_voxels) * 100
    
    val_tumor_percentages.append(tumor_pct)
    val_tumor_voxels.append(tumor_voxels)
    
    print(f"\nBatch {i+1}:")
    print(f"  Shape: {image.shape}")
    print(f"  Tumor voxels: {tumor_voxels:,} / {total_voxels:,}")
    print(f"  Tumor %: {tumor_pct:.2f}%")
    
    if tumor_pct < 1.0:
        print(f"  ❌ CRITICAL: Validation tumor % too low!")
    elif tumor_pct < 5.0:
        print(f"  ⚠️  Warning: Low validation tumor %")
    else:
        print(f"  ✅ Good validation tumor %")

avg_val_tumor = np.mean(val_tumor_percentages)
print(f"\n📈 Validation Data Summary:")
print(f"  Average tumor %: {avg_val_tumor:.2f}%")
print(f"  Min tumor %: {min(val_tumor_percentages):.2f}%")
print(f"  Max tumor %: {max(val_tumor_percentages):.2f}%")

# =============================================================================
# 3. COMPARE TRAIN vs VAL
# =============================================================================
print("\n" + "="*70)
print("⚖️  TRAIN vs VALIDATION COMPARISON:")
print("-"*70)
print(f"Training avg tumor %:   {avg_train_tumor:.2f}%")
print(f"Validation avg tumor %: {avg_val_tumor:.2f}%")
print(f"Ratio (should be ~1.0): {avg_train_tumor / max(avg_val_tumor, 0.01):.2f}x")

if abs(avg_train_tumor - avg_val_tumor) > 2.0:
    print("\n❌ PROBLEM DETECTED:")
    print("   Train and validation have very different tumor percentages!")
    print("   This will cause unreliable validation metrics.")
    print("\n🔧 FIX:")
    print("   Make sure BOTH train and val transforms use:")
    print("   RandCropByPosNegLabeld with same pos/neg ratio")
else:
    print("\n✅ Train/Val tumor percentages are balanced!")

# =============================================================================
# 4. DATASET SIZE CHECK
# =============================================================================
print("\n" + "="*70)
print("📦 DATASET SIZE:")
print("-"*70)
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Batch size (train): {next(iter(train_loader))['image'].shape[0]}")
print(f"Batch size (val): {next(iter(val_loader))['image'].shape[0]}")

# =============================================================================
# 5. READINESS ASSESSMENT
# =============================================================================
print("\n" + "="*70)
print("🎯 TRAINING READINESS ASSESSMENT:")
print("="*70)

issues = []
warnings = []

if avg_train_tumor < 1.0:
    issues.append("Training tumor % < 1% - will not learn")
elif avg_train_tumor < 5.0:
    warnings.append(f"Training tumor % is low ({avg_train_tumor:.2f}%) - consider increasing pos/neg ratio")
else:
    print("✅ Training tumor percentage: GOOD")

if avg_val_tumor < 1.0:
    issues.append("Validation tumor % < 1% - metrics unreliable")
elif avg_val_tumor < 5.0:
    warnings.append(f"Validation tumor % is low ({avg_val_tumor:.2f}%)")
else:
    print("✅ Validation tumor percentage: GOOD")

if abs(avg_train_tumor - avg_val_tumor) > 3.0:
    issues.append(f"Train ({avg_train_tumor:.2f}%) and Val ({avg_val_tumor:.2f}%) tumor % mismatch")
else:
    print("✅ Train/Val balance: GOOD")

if len(train_loader) < 10:
    warnings.append(f"Only {len(train_loader)} training batches - consider num_samples or batch_size")
else:
    print(f"✅ Training batches: {len(train_loader)}")

# Print summary
print("\n" + "-"*70)
if issues:
    print("❌ CRITICAL ISSUES (must fix before training):")
    for issue in issues:
        print(f"   • {issue}")
    print("\n🔧 RECOMMENDED FIXES:")
    print("   1. Increase pos/neg ratio to 5:1 or 7:1")
    print("   2. Make sure both train AND val use RandCropByPosNegLabeld")
    print("   3. Decrease spatial_size if needed (e.g., 64→48)")
else:
    print("✅ No critical issues!")

if warnings:
    print("\n⚠️  WARNINGS (consider addressing):")
    for warning in warnings:
        print(f"   • {warning}")
else:
    print("✅ No warnings!")

if not issues:
    print("\n" + "="*70)
    print("🚀 READY TO TRAIN!")
    print("="*70)
    print("\nExpected results:")
    print(f"  • Dice score should reach: 0.3-0.6 (with {avg_train_tumor:.1f}% tumor)")
    print(f"  • Training will be: {'Slow' if avg_train_tumor < 5 else 'Normal'}")
    print(f"  • Validation metrics: {'Reliable' if abs(avg_train_tumor - avg_val_tumor) < 2 else 'May fluctuate'}")
else:
    print("\n" + "="*70)
    print("⚠️  NOT READY - FIX ISSUES FIRST")
    print("="*70)
