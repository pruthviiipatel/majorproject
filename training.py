## Cell 8: Simple U-Net for Binary Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryUNet(nn.Module):
    """Simple U-Net specifically for binary segmentation"""
    def __init__(self):
        super(BinaryUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(1, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._block(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)
        
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        
        # Final layer - SINGLE OUTPUT CHANNEL
        self.final = nn.Conv3d(32, 1, kernel_size=1)
        
        self.pool = nn.MaxPool3d(2)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# Create model
model = BinaryUNet().to(device)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"✓ Output channels: {model.final.out_channels}")  # Should be 1

import torch
import torch.nn as nn
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandRotate90d, RandFlipd,
    RandScaleIntensityd, RandShiftIntensityd, EnsureTyped
)
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss, FocalLoss

print("="*70)
print("🔥 NUCLEAR OPTION: MICROSCOPIC TUMOR CONFIGURATION")
print("="*70)
print("Your dataset stats:")
print("  Average tumor: 0.028% (EXTREMELY SMALL)")
print("  This requires the most aggressive settings possible!")
print()

# =============================================================================
# EXTREME PARAMETERS
# =============================================================================
SPATIAL_SIZE = (32, 32, 32)   # VERY SMALL patches to concentrate tumor
POS_NEG_RATIO = 19            # 19:1 = 95% tumor patches!
NUM_SAMPLES_TRAIN = 8         # Many samples to find those tiny tumors
NUM_SAMPLES_VAL = 6
BATCH_SIZE = 8                # Higher batch size for stability

print(f"Configuration:")
print(f"  Spatial size: {SPATIAL_SIZE} (very small to concentrate tumor)")
print(f"  Pos:Neg ratio: {POS_NEG_RATIO}:1 (95% of patches contain tumor!)")
print(f"  Samples per volume: {NUM_SAMPLES_TRAIN} (train) / {NUM_SAMPLES_VAL} (val)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Expected tumor % in batches: 20-30%")
print()

# =============================================================================
# TRANSFORMS
# =============================================================================
base_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.5, 1.5, 2.0),
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
        margin=5,  # Smaller margin
    ),
]

# Extreme tumor-focused cropping
tumor_crop_train = RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=SPATIAL_SIZE,
    pos=POS_NEG_RATIO,  # 95% tumor!
    neg=1,
    num_samples=NUM_SAMPLES_TRAIN,
    image_key="image",
    image_threshold=0,
    allow_smaller=False,
)

tumor_crop_val = RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=SPATIAL_SIZE,
    pos=POS_NEG_RATIO,  # Same as training
    neg=1,
    num_samples=NUM_SAMPLES_VAL,
    image_key="image",
    image_threshold=0,
    allow_smaller=False,
)

# Training transforms
train_transforms = Compose(
    base_transforms + [
        tumor_crop_train,
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.5),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
)

# Validation transforms
val_transforms = Compose(
    base_transforms + [
        tumor_crop_val,
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
)

# Create data dictionaries
data_dicts = []
for img_name, label_name in zip(image_files, label_files):
    data_dicts.append({
        "image": os.path.join(IMAGES_DIR, img_name),
        "label": os.path.join(LABELS_DIR, label_name),
    })

# Split into train and validation
train_size = int(0.8 * len(data_dicts))
val_size = len(data_dicts) - train_size

train_files = data_dicts[:train_size]
val_files = data_dicts[train_size:]

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# Create datasets
# Recreate datasets with FIXED transforms
from monai.data import CacheDataset, DataLoader

# =============================================================================
# CREATE DATASETS
# =============================================================================
print("\n🔄 Creating datasets...")

print("🔄 Creating datasets with EXTREME tumor focus...")

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=4,
)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=4,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

print(f"✓ Training batches: {len(train_loader)}")
print(f"✓ Validation batches: {len(val_loader)}")

# NOW RUN THE VERIFICATION SCRIPT
exec(open('verify_data.py').read())


print("\nData loaders created successfully!")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

train_batch = next(iter(train_loader))
if isinstance(train_batch, list):
    train_batch = train_batch[0]
val_batch = next(iter(val_loader))

train_tumor_pct = (train_batch["label"] > 0).sum().item() / train_batch["label"].numel() * 100
val_tumor_pct = (val_batch["label"] > 0).sum().item() / val_batch["label"].numel() * 100

print(f"Train tumor %: {train_tumor_pct:.2f}%")
print(f"Val tumor %:   {val_tumor_pct:.2f}%")

if train_tumor_pct > 15 and val_tumor_pct > 15:
    print("\n✅ EXCELLENT! Successfully concentrated tumors!")
    print(f"   From 0.028% in raw data → {train_tumor_pct:.1f}% in training batches")
    print(f"   That's a {train_tumor_pct/0.028:.0f}x concentration!")
elif train_tumor_pct > 10:
    print("\n✅ GOOD! Tumor concentration achieved")
else:
    print("\n⚠️  Still low. Your tumors are REALLY tiny.")
    print("   May need to filter out files with <1000 tumor voxels")

## 🔍 DEBUG: Check Your Data (CORRECTED)

import torch

# Check training batch
batch = next(iter(train_loader))

# Training data returns a LIST, extract first element
if isinstance(batch, list):
    batch = batch[0]

image = batch["image"]
label = batch["label"]

print("="*70)
print("TRAINING DATA")
print("="*70)
print(f"Image shape: {image.shape}")
print(f"Image range: {image.min():.4f} to {image.max():.4f}")
print(f"\nLabel shape: {label.shape}")
print(f"Unique label values: {torch.unique(label).cpu().numpy()}")
print(f"Tumor voxels: {(label > 0).sum().item()}")
print(f"Tumor %: {(label > 0).sum().item() / label.numel() * 100:.2f}%")

# Check validation
val_batch = next(iter(val_loader))
val_label = val_batch["label"]

print(f"\n{'='*70}")
print("VALIDATION DATA")
print("="*70)
print(f"Val label unique values: {torch.unique(val_label).cpu().numpy()}")
print(f"Val tumor %: {(val_label > 0).sum().item() / val_label.numel() * 100:.2f}%")
print("="*70)

## Quick Check - Does training data have tumor?

batch = next(iter(train_loader))
if isinstance(batch, list):
    batch = batch[0]
    
label = batch["label"]
print(f"Unique values: {torch.unique(label).cpu().numpy()}")
print(f"Tumor %: {(label > 0).sum().item() / label.numel() * 100:.4f}%")
print(f"Tumor voxels: {(label > 0).sum().item()} / {label.numel()}")

class MicroscopicTumorLoss(nn.Module):
    """
    Specialized loss for datasets with 0.02-0.05% tumor
    Extremely aggressive weighting
    """
    def __init__(self):
        super().__init__()
        
        # Dice - focus on any overlap at all
        self.dice = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        
        # Focal - VERY aggressive
        self.focal = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            alpha=0.25,
            gamma=4.0,  # Very high gamma
        )
        
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        
        # EXTREME weighting - 50x on focal
        return dice_loss + (50.0 * focal_loss)

loss_function = MicroscopicTumorLoss()
print("✓ Loss: Dice + 50x Focal (gamma=4.0)")
print("  This will HEAVILY penalize missing any tumor voxels")

# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================
print("\n" + "="*70)
print("⚙️  OPTIMIZER CONFIGURATION")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming model is already defined
model = model.to(device)
loss_function = loss_function.to(device)

# Higher learning rate for faster learning
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-3,  # Even higher than before
    weight_decay=1e-5,
    betas=(0.9, 0.999),
)

# Aggressive scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,
    eta_min=1e-6,
)

print(f"✓ Optimizer: AdamW (lr=2e-3)")
print(f"✓ Scheduler: CosineAnnealingWarmRestarts")
print(f"✓ Device: {device}")

## Cell 12: Simple Training Loop

from tqdm import tqdm
from monai.inferers import sliding_window_inference

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch_data in tqdm(loader, desc="Training"):
        if isinstance(batch_data, list):
            batch_data = batch_data[0]
        
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, loss_fn, metric, device):
    model.eval()
    total_loss = 0
    metric.reset()
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Validating"):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = sliding_window_inference(inputs, (96,96,96), 1, model)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Binary prediction
            preds = (torch.sigmoid(outputs) > 0.5).float()
            metric(y_pred=preds, y=labels)
    
    return total_loss / len(loader), metric.aggregate().item()

# Train
num_epochs = 50
best_dice = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
    val_loss, val_dice = validate(model, val_loader, loss_function, dice_metric, device)
    
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Dice: {val_dice:.4f}")
    
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Best model saved!")
