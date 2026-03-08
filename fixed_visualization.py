"""
FIXED VISUALIZATION CODE
Handles dimension mismatches and finds slices with actual tumor
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# DEBUG: Check shapes first
# =============================================================================
print("="*70)
print("🔍 DEBUGGING SHAPES")
print("="*70)
print(f"Original image shape: {original_image.shape}")
print(f"Ground truth shape: {gt_label.shape}")
print(f"Predicted mask shape: {predicted_mask.shape}")
print()

# Fix dimension issues
if len(predicted_mask.shape) == 4:
    # If 4D, squeeze extra dimension
    predicted_mask = predicted_mask.squeeze()
    print(f"✓ Squeezed prediction to: {predicted_mask.shape}")

if len(predicted_mask.shape) == 2:
    # If 2D, something went very wrong
    print("❌ ERROR: Prediction is 2D! Check inference code.")
    
# Ensure all have same shape
if predicted_mask.shape != gt_label.shape:
    print(f"\n⚠️  WARNING: Shape mismatch!")
    print(f"Ground truth: {gt_label.shape}")
    print(f"Prediction: {predicted_mask.shape}")
    print("Visualization may not work correctly.")

# =============================================================================
# FIND SLICES WITH TUMOR
# =============================================================================
print("\n" + "="*70)
print("🔍 FINDING SLICES WITH TUMOR")
print("="*70)

# Find slices that contain tumor in ground truth
tumor_slices = []
for i in range(gt_label.shape[2]):
    if (gt_label[:, :, i] > 0).any():
        tumor_count = (gt_label[:, :, i] > 0).sum()
        tumor_slices.append((i, tumor_count))

if tumor_slices:
    # Sort by tumor content
    tumor_slices.sort(key=lambda x: x[1], reverse=True)
    print(f"\nFound {len(tumor_slices)} slices with tumor")
    print(f"Top 5 slices by tumor content:")
    for i, (slice_idx, count) in enumerate(tumor_slices[:5]):
        print(f"  {i+1}. Slice {slice_idx}: {count} tumor voxels")
    
    # Select slices: max tumor, middle, min tumor
    if len(tumor_slices) >= 3:
        selected_slices = [
            tumor_slices[0][0],  # Most tumor
            tumor_slices[len(tumor_slices)//2][0],  # Medium
            tumor_slices[-1][0],  # Least tumor (but still has some)
        ]
    else:
        selected_slices = [s[0] for s in tumor_slices]
    
    print(f"\nSelected slices for visualization: {selected_slices}")
else:
    print("\n⚠️  No tumor found in ground truth!")
    print("Using default slices (may show empty regions)")
    d = gt_label.shape[2]
    selected_slices = [d//4, d//2, 3*d//4]

# =============================================================================
# IMPROVED VISUALIZATION FUNCTION
# =============================================================================
def visualize_prediction(image, ground_truth, prediction, slice_idx):
    """
    Compare ground truth and predicted segmentation
    Handles dimension mismatches and empty predictions
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Get the slice
    img_slice = image[:, :, slice_idx]
    gt_slice = ground_truth[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]
    
    # Count voxels
    gt_tumor = (gt_slice > 0).sum()
    pred_tumor = (pred_slice > 0).sum()
    
    # Original CT scan
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title(f'Original CT Scan\nSlice {slice_idx}', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth overlay
    axes[0, 1].imshow(img_slice, cmap='gray')
    if gt_tumor > 0:
        axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[0, 1].set_title(f'Ground Truth Overlay\n{gt_tumor} tumor voxels', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Predicted overlay
    axes[0, 2].imshow(img_slice, cmap='gray')
    if pred_tumor > 0:
        axes[0, 2].imshow(pred_slice, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    else:
        axes[0, 2].text(0.5, 0.5, 'NO TUMOR\nPREDICTED', 
                       ha='center', va='center', 
                       transform=axes[0, 2].transAxes,
                       fontsize=20, color='red', fontweight='bold')
    axes[0, 2].set_title(f'Predicted Overlay\n{pred_tumor} tumor voxels', 
                         fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ground truth mask only
    axes[1, 0].imshow(gt_slice, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Predicted mask only
    axes[1, 1].imshow(pred_slice, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Difference map (TP, FP, FN)
    # True Positive (both predict tumor) - Green
    # False Positive (predict tumor, no GT) - Red
    # False Negative (GT tumor, no prediction) - Yellow
    diff_map = np.zeros((*gt_slice.shape, 3))
    
    # True Positives - Green
    tp_mask = (gt_slice > 0) & (pred_slice > 0)
    diff_map[tp_mask] = [0, 1, 0]
    
    # False Positives - Red
    fp_mask = (gt_slice == 0) & (pred_slice > 0)
    diff_map[fp_mask] = [1, 0, 0]
    
    # False Negatives - Yellow
    fn_mask = (gt_slice > 0) & (pred_slice == 0)
    diff_map[fn_mask] = [1, 1, 0]
    
    axes[1, 2].imshow(diff_map)
    
    # Calculate metrics for this slice
    tp = tp_mask.sum()
    fp = fp_mask.sum()
    fn = fn_mask.sum()
    
    if gt_tumor > 0:
        slice_dice = (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
        title = f'Difference Map\nDice: {slice_dice:.3f}'
    else:
        title = 'Difference Map\n(No tumor in GT)'
    
    axes[1, 2].set_title(title, fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'True Pos: {tp}'),
        Patch(facecolor='red', label=f'False Pos: {fp}'),
        Patch(facecolor='yellow', label=f'False Neg: {fn}')
    ]
    axes[1, 2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

# =============================================================================
# VISUALIZE SELECTED SLICES
# =============================================================================
print("\n" + "="*70)
print("🎨 CREATING VISUALIZATIONS")
print("="*70)

# Create visualization directory if it doesn't exist
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

for i, slice_num in enumerate(selected_slices):
    print(f"\nGenerating visualization {i+1}/{len(selected_slices)} (slice {slice_num})...")
    
    fig = visualize_prediction(original_image, gt_label, predicted_mask, slice_idx=slice_num)
    
    # Save
    save_path = os.path.join(VIZ_DIR, f'prediction_slice_{slice_num:03d}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    plt.show()
    plt.close()

# =============================================================================
# OVERALL STATISTICS
# =============================================================================
print("\n" + "="*70)
print("📊 OVERALL PREDICTION STATISTICS")
print("="*70)

total_gt_tumor = (gt_label > 0).sum()
total_pred_tumor = (predicted_mask > 0).sum()

print(f"\nGround truth tumor voxels: {total_gt_tumor:,}")
print(f"Predicted tumor voxels: {total_pred_tumor:,}")

if total_pred_tumor == 0:
    print("\n⚠️  MODEL PREDICTED NO TUMOR AT ALL!")
    print("This is expected with:")
    print("  - Very early training (low epochs)")
    print("  - Extremely small tumors (0.028% in your dataset)")
    print("  - Model hasn't learned to detect tumors yet")
    print("\n💡 Continue training with nuclear configuration!")
else:
    print(f"\n✓ Model is detecting tumors!")
    
    # Calculate overall Dice
    intersection = ((gt_label > 0) & (predicted_mask > 0)).sum()
    overall_dice = (2.0 * intersection) / (total_gt_tumor + total_pred_tumor)
    
    print(f"Overall Dice score: {overall_dice:.4f}")
    
    if overall_dice < 0.1:
        print("⚡ Low Dice - model is learning but needs more training")
    elif overall_dice < 0.3:
        print("⚡ Moderate Dice - making progress!")
    else:
        print("✅ Good Dice - model is working well!")

print("\n" + "="*70)
print("✅ Visualization complete!")
print(f"Saved {len(selected_slices)} images to {VIZ_DIR}/")
print("="*70)
