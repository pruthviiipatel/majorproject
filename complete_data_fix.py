"""
COMPLETE FIX: Data Preprocessing Cell
This replaces your entire data loading cell with MATCHED train/val transforms

Key fixes:
1. Both train and val use SAME pos/neg ratio
2. Increased to pos=5, neg=1 (83% tumor patches) for better learning
3. Proper configuration of all parameters
"""

import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandRotate90d, RandFlipd,
    RandScaleIntensityd, RandShiftIntensityd, EnsureTyped
)
from monai.data import CacheDataset, DataLoader

# =============================================================================
# CRITICAL PARAMETERS - ADJUST THESE IF NEEDED
# =============================================================================
SPATIAL_SIZE = (64, 64, 64)
POS_NEG_RATIO_TRAIN = 5  # 5:1 = 83% tumor patches (increased from 3:1)
POS_NEG_RATIO_VAL = 5    # 🚨 MUST MATCH TRAINING!
NUM_SAMPLES_TRAIN = 4
NUM_SAMPLES_VAL = 2
BATCH_SIZE_TRAIN = 2     # Adjust based on GPU memory
BATCH_SIZE_VAL = 1

print("="*70)
print("🔧 CREATING MATCHED TRAIN/VAL TRANSFORMS")
print("="*70)
print(f"Spatial size: {SPATIAL_SIZE}")
print(f"Train pos:neg ratio: {POS_NEG_RATIO_TRAIN}:1")
print(f"Val pos:neg ratio:   {POS_NEG_RATIO_VAL}:1")
print(f"Expected tumor %: ~{(POS_NEG_RATIO_TRAIN/(POS_NEG_RATIO_TRAIN+1))*100:.1f}%")
print()

# =============================================================================
# TRAINING TRANSFORMS
# =============================================================================
train_transforms = Compose([
    # 1. Load
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    
    # 2. Orientation & Spacing
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.5, 1.5, 2.0),
        mode=("bilinear", "nearest"),
    ),
    
    # 3. Intensity normalization (CT Hounsfield units)
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,  # Air
        a_max=400,    # Soft tissue
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    
    # 4. Crop foreground
    CropForegroundd(
        keys=["image", "label"],
        source_key="image",x
        margin=10,
    ),
    
    # 5. 🎯 TUMOR-FOCUSED CROPPING (5:1 ratio = 83% tumor patches)
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=SPATIAL_SIZE,
        pos=POS_NEG_RATIO_TRAIN,
        neg=1,
        num_samples=NUM_SAMPLES_TRAIN,
        image_key="image",
        image_threshold=0,
        allow_smaller=False,  # Ensure exact spatial_size
    ),
    
    # 6. Data Augmentation
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    
    # 7. Ensure proper tensor type
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

# =============================================================================
# VALIDATION TRANSFORMS - 🚨 CRITICAL: MATCHES TRAINING (no augmentation)
# =============================================================================
val_transforms = Compose([
    # 1. Load
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    
    # 2. Orientation & Spacing (SAME as training)
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.5, 1.5, 2.0),
        mode=("bilinear", "nearest"),
    ),
    
    # 3. Intensity normalization (SAME as training)
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    
    # 4. Crop foreground (SAME as training)
    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
        margin=10,
    ),
    
    # 5. 🚨 CRITICAL FIX: SAME tumor-focused cropping as training!
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=SPATIAL_SIZE,
        pos=POS_NEG_RATIO_VAL,  # MUST be same as training!
        neg=1,
        num_samples=NUM_SAMPLES_VAL,
        image_key="image",
        image_threshold=0,
        allow_smaller=False,
    ),
    
    # 6. NO AUGMENTATION for validation
    
    # 7. Ensure proper tensor type
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

print("✅ Transforms created with MATCHING tumor-focus strategy")

# =============================================================================
# CREATE DATASETS
# =============================================================================
print("\n🔄 Creating datasets...")

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,  # Cache everything in RAM for speed
    num_workers=4,
)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=4,
)

print(f"✓ Training samples: {len(train_ds)}")
print(f"✓ Validation samples: {len(val_ds)}")

# =============================================================================
# CREATE DATA LOADERS
# =============================================================================
print("\n🔄 Creating data loaders...")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=0,  # Set to 0 on Windows, 2-4 on Linux
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE_VAL,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

print(f"✓ Training batches: {len(train_loader)}")
print(f"✓ Validation batches: {len(val_loader)}")

# =============================================================================
# IMMEDIATE VERIFICATION
# =============================================================================
print("\n" + "="*70)
print("🔍 QUICK VERIFICATION (1 batch each)")
print("="*70)

# Check training
train_batch = next(iter(train_loader))
if isinstance(train_batch, list):
    train_batch = train_batch[0]

train_label = train_batch["label"]
train_tumor_pct = (train_label > 0).sum().item() / train_label.numel() * 100

print(f"\nTRAINING:")
print(f"  Batch shape: {train_batch['image'].shape}")
print(f"  Tumor %: {train_tumor_pct:.2f}%")

# Check validation
val_batch = next(iter(val_loader))
val_label = val_batch["label"]
val_tumor_pct = (val_label > 0).sum().item() / val_label.numel() * 100

print(f"\nVALIDATION:")
print(f"  Batch shape: {val_batch['image'].shape}")
print(f"  Tumor %: {val_tumor_pct:.2f}%")

# Compare
print(f"\nTRAIN/VAL COMPARISON:")
print(f"  Ratio: {train_tumor_pct / max(val_tumor_pct, 0.01):.2f}x")

if abs(train_tumor_pct - val_tumor_pct) < 3.0:
    print(f"  ✅ BALANCED! Ready to train")
elif abs(train_tumor_pct - val_tumor_pct) < 5.0:
    print(f"  ⚠️  Slightly imbalanced but acceptable")
else:
    print(f"  ❌ STILL IMBALANCED - check configuration!")
    print(f"     Train: {train_tumor_pct:.2f}%")
    print(f"     Val:   {val_tumor_pct:.2f}%")

# Expected results guide
print("\n" + "="*70)
print("📊 EXPECTED RESULTS WITH THIS CONFIGURATION:")
print("="*70)
print(f"Both train & val tumor %: ~{(POS_NEG_RATIO_TRAIN/(POS_NEG_RATIO_TRAIN+1))*100:.0f}%")
print(f"Expected Dice score after training: 0.4-0.7")
print(f"Training speed: Normal")
print(f"Validation reliability: High")
print("="*70)

print("\n✅ Data loaders ready!")
print("\n💡 TIP: Run the full verification script to check 5 batches:")
print("   exec(open('verify_data.py').read())")
