import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
from tqdm import tqdm
import os
import datetime
from sklearn.model_selection import train_test_split
import pandas as pd

# Image Preprocessing Functions
# Preprocessing with nan handling
def preprocess_image(image):
    """
    Perform preprocessing on image:
    1. Grayscale conversion
    2. Normalization
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Gamma transformation
    
    Args:
        image: Input image (numpy array or PIL image)
    Returns:
        Preprocessed image
    """
    # Convert to numpy if PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if RGB (using the formula from paper)
    if len(image.shape) == 3:
        # Using the formula: I_gray = 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        image = gray.astype(np.uint8)
    
    # Normalize with nan handling
    mean = np.mean(image)
    std = np.std(image)
    
    # Handle zero or very small std to prevent division by zero
    if std < 1e-5:
        norm_img = np.zeros_like(image, dtype=float)
    else:
        norm_img = (image - mean) / std
    
    # Replace inf/nan values
    norm_img = np.nan_to_num(norm_img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to uint8 for CLAHE (safely)
    norm_min = norm_img.min()
    norm_max = norm_img.max()
    
    # Handle case where min == max
    if np.abs(norm_max - norm_min) < 1e-5:
        norm_scaled = np.zeros_like(norm_img, dtype=np.uint8)
    else:
        norm_scaled = ((norm_img - norm_min) * 255 / (norm_max - norm_min)).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(norm_scaled)
    
    # Apply gamma transformation (optional, gamma=1.2 for example)
    gamma = 1.2
    gamma_corrected = np.power(enhanced_img / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    return gamma_corrected

# Enhanced Dataset with Preprocessing
class EnhancedVeinDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, transform=None, slice_size=48):
        """
        Args:
            csv_file (str): Path to CSV with BatID and ImageID columns
            image_dir (str): Path to image folder
            mask_dir (str): Path to mask folder
            transform (callable, optional): Optional transform to be applied
            slice_size (int): Size of image patches to slice
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.slice_size = slice_size
        
        # Generate sliced patches
        self.patches = self._generate_patches()
        
    def _generate_patches(self):
        """Generate patches from original images for data augmentation"""
        patches = []
        for idx in range(len(self.data)):
            # Get image ID
            image_id = self.data.iloc[idx]['ImageID']
            
            # Load image and mask
            img_path = os.path.join(self.image_dir, f"{image_id}.png")
            mask_path = os.path.join(self.mask_dir, f"{image_id}.jpg")
            
            # Open and preprocess image
            image = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            
            # Get image dimensions
            width, height = image.size
            
            # Generate random patches (as in the paper Fig. 3)
            # Number of patches per image - more patches for training
            num_patches = 20  # Adjust as needed
            
            for _ in range(num_patches):
                # Random center point within image
                x = np.random.randint(self.slice_size//2, width - self.slice_size//2)
                y = np.random.randint(self.slice_size//2, height - self.slice_size//2)
                
                # Crop patches
                img_patch = image.crop((x - self.slice_size//2, y - self.slice_size//2,
                                      x + self.slice_size//2, y + self.slice_size//2))
                mask_patch = mask.crop((x - self.slice_size//2, y - self.slice_size//2,
                                       x + self.slice_size//2, y + self.slice_size//2))
                
                patches.append((image_id, img_patch, mask_patch))
        
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image_id, image_patch, mask_patch = self.patches[idx]
        
        # Preprocess image
        preprocessed_img = preprocess_image(image_patch)
        preprocessed_img = Image.fromarray(preprocessed_img)
        
        # Convert to tensor and normalize
        image = TF.to_tensor(preprocessed_img)
        mask = TF.to_tensor(mask_patch)
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Double Convolution Block - Same as in your original U-Net
# Double Convolution Block with adaptive normalization based on spatial dimensions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, min_spatial_dim=2):
        """
        Double convolution block with adaptive normalization.
        Uses BatchNorm for larger feature maps and GroupNorm for smaller ones.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            min_spatial_dim: Minimum spatial dimension to use BatchNorm (otherwise use GroupNorm)
        """
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # We'll choose normalization based on the output spatial dimensions
        # (determined at forward time)
        self.norm1_bn = nn.BatchNorm2d(out_channels)
        # For GroupNorm, use 8 groups or out_channels if smaller
        num_groups = min(8, out_channels)
        self.norm1_gn = nn.GroupNorm(num_groups, out_channels)
        
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2_bn = nn.BatchNorm2d(out_channels)
        self.norm2_gn = nn.GroupNorm(num_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.min_spatial_dim = min_spatial_dim

    def forward(self, x):
        # First convolution
        x = self.conv1(x)
        
        # Choose normalization based on spatial dimensions
        if x.shape[2] >= self.min_spatial_dim and x.shape[3] >= self.min_spatial_dim:
            # Use BatchNorm for larger feature maps
            x = self.norm1_bn(x)
        else:
            # Use GroupNorm for smaller feature maps
            x = self.norm1_gn(x)
            
        x = self.relu1(x)
        
        # Second convolution
        x = self.conv2(x)
        
        # Choose normalization again
        if x.shape[2] >= self.min_spatial_dim and x.shape[3] >= self.min_spatial_dim:
            x = self.norm2_bn(x)
        else:
            x = self.norm2_gn(x)
            
        x = self.relu2(x)
        
        return x

# Weighted Feature Fusion for Bi-FPN
class WeightedFeatureFusion(nn.Module):
    def __init__(self, num_inputs, epsilon=0.0001):
        super(WeightedFeatureFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, inputs):
        # Apply ReLU to weights
        weights = F.relu(self.weights)
        
        # Normalize weights
        norm_weights = weights / (weights.sum() + self.epsilon)
        
        # Apply weights to inputs and sum
        # Ensure all inputs have same shape before weighted fusion
        # In this case, they should, but just to be safe:
        if not all(inp.shape == inputs[0].shape for inp in inputs):
            raise ValueError(f"Input shapes don't match: {[inp.shape for inp in inputs]}")
            
        # Weighted sum
        out = sum(input_tensor * w for input_tensor, w in zip(inputs, norm_weights))
        
        return out

# Simplified BiFPN implementation that directly works with U-Net skip connections
class BiFPNBlock(nn.Module):
    def __init__(self, feature_dims):
        """
        Args:
            feature_dims: List of channel dimensions at each level (e.g. [32, 64, 128, 256, 512])
                         Listed from highest resolution to lowest resolution
        """
        super(BiFPNBlock, self).__init__()
        self.feature_dims = feature_dims
        
        # For top-down path: project from higher level (fewer channels) to lower level (more channels)
        # For U-Net skip connections in the order [fine_level, ..., coarse_level]
        # We need to go from coarse_level -> fine_level in top-down path
        
        # Create a projection for each pair of adjacent levels
        self.td_projections = nn.ModuleList()
        for i in range(len(feature_dims) - 1):
            # Project from level i+1 to level i (more channels to fewer channels)
            self.td_projections.append(
                nn.Conv2d(feature_dims[i+1], feature_dims[i], kernel_size=1)
            )
        
        # For bottom-up path: project from lower level (more channels) to higher level (fewer channels)
        self.bu_projections = nn.ModuleList()
        for i in range(len(feature_dims) - 1):
            # Project from level i to level i+1 (fewer channels to more channels)
            self.bu_projections.append(
                nn.Conv2d(feature_dims[i], feature_dims[i+1], kernel_size=1)
            )
        
        # Convolutions after fusion for both paths
        self.td_convs = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        
        for dim in feature_dims:
            # Process each feature level after fusion
            self.td_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                )
            )
            
            self.bu_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Weighted fusion for each level
        self.td_fusions = nn.ModuleList([WeightedFeatureFusion(2) for _ in range(len(feature_dims))])
        self.bu_fusions = nn.ModuleList([WeightedFeatureFusion(2) for _ in range(len(feature_dims))])

    def forward(self, features):
        """
        Args:
            features: list of features from encoder path, ordered from highest resolution 
                     to lowest resolution [fine_level, ..., coarse_level]
        Returns:
            Enhanced features with same ordering and dimensions
        """
        num_levels = len(features)
        
        # Store intermediate features
        td_features = [None] * num_levels  # Top-down path
        bu_features = [None] * num_levels  # Bottom-up path
        
        # Top-down path (coarse -> fine)
        # Start from the coarsest level
        td_features[num_levels - 1] = features[num_levels - 1]
        
        # Process remaining levels from coarse to fine
        for i in range(num_levels - 2, -1, -1):
            # Get higher level feature and project to current dimension
            higher_feature = td_features[i + 1]
            projected_higher = self.td_projections[i](higher_feature)
            
            # Resize to current feature map size
            resized_higher = F.interpolate(
                projected_higher, 
                size=features[i].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            
            # Fuse with current level feature
            fused = self.td_fusions[i]([features[i], resized_higher])
            td_features[i] = self.td_convs[i](fused)
        
        # Bottom-up path (fine -> coarse)
        # Start from the finest level
        bu_features[0] = td_features[0]
        
        # Process remaining levels from fine to coarse
        for i in range(1, num_levels):
            # Get lower level feature and project to current dimension
            lower_feature = bu_features[i - 1]
            projected_lower = self.bu_projections[i - 1](lower_feature)
            
            # Downsample to current feature map size
            if projected_lower.shape[2:] != features[i].shape[2:]:
                resized_lower = F.adaptive_max_pool2d(
                    projected_lower, 
                    output_size=features[i].shape[2:]
                )
            else:
                resized_lower = projected_lower
            
            # Fuse with current level feature
            fused = self.bu_fusions[i]([td_features[i], resized_lower])
            bu_features[i] = self.bu_convs[i](fused)
        
        return bu_features

# Improved U-Net with Bi-FPN and adaptive normalization
class UNetBiFPN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], max_levels=None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Feature dimensions at each level
            max_levels: Maximum number of downsampling levels to use (to prevent too small feature maps)
        """
        super(UNetBiFPN, self).__init__()
        
        # If max_levels is provided, limit the feature levels
        if max_levels is not None and max_levels < len(features):
            features = features[:max_levels]
            
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features = features
        
        # Downsampling/Encoder path
        in_feat = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_feat, feature))
            in_feat = feature
        
        # Bottleneck - use GroupNorm for the bottleneck to avoid BatchNorm issues
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Feature fusion with Bi-FPN
        self.bifpn = BiFPNBlock(feature_dims=features)
        
        # Upsampling/Decoder path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Save original input dimensions to check how much we can downsample
        orig_h, orig_w = x.shape[2], x.shape[3]
        min_dim = min(orig_h, orig_w)
        
        # Calculate max possible downsampling levels (prevent feature maps < 2x2)
        max_downsample = int(np.log2(min_dim)) - 1  # Leave at least 2x2
        actual_levels = min(len(self.downs), max_downsample)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder path - only go as deep as our input size allows
        for i in range(actual_levels):
            x = self.downs[i](x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Make sure skip_connections list has the right number of levels
        skip_connections = skip_connections[:actual_levels]
        
        # Apply Bi-FPN to enhance skip connections, maintaining their original order
        enhanced_skips = self.bifpn(skip_connections)
        
        # Reversed for decoder path (coarse to fine)
        enhanced_skips = enhanced_skips[::-1]
        
        # Decoder path with enhanced skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            
            # Get corresponding skip connection
            skip_idx = idx // 2
            if skip_idx < len(enhanced_skips):
                skip = enhanced_skips[skip_idx]
                
                # Handle case where sizes don't match exactly
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
                # Concatenate with enhanced skip connection
                concat_skip = torch.cat((skip, x), dim=1)
                
                # Double convolution
                x = self.ups[idx+1](concat_skip)
            else:
                # If we ran out of skip connections, just use upsampled x
                x = self.ups[idx+1](torch.cat((x, x), dim=1))
        
        # Final 1x1 convolution and sigmoid
        return self.sigmoid(self.final_conv(x))

# Loss functions from the paper
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.2, bce_weight=0.8):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        
    def forward(self, predictions, targets):
        dice_loss = self.dice(predictions, targets)
        bce_loss = self.bce(predictions, targets)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

# Helper function to create train/validation/test splits while keeping bats together
def create_train_val_test_splits_enhanced(csv_file, image_dir, mask_dir, test_size=0.15, val_size=0.15, batch_size=2, random_state=420):
    """
    Create train, validation and test splits while keeping all images from the same bat together.
    Uses enhanced dataset with preprocessing.
    """
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Get unique bat IDs
    unique_bats = df['BatID'].unique()
    
    # First split off the test set
    train_val_bats, test_bats = train_test_split(
        unique_bats, 
        test_size=test_size,
        random_state=random_state
    )
    
    # Then split the remaining data into train and validation
    train_bats, val_bats = train_test_split(
        train_val_bats,
        test_size=val_size,
        random_state=random_state
    )
    
    # Create full dataset with enhanced preprocessing
    full_dataset = EnhancedVeinDataset(csv_file, image_dir, mask_dir)
    
    # Get indices for each split
    train_indices = df[df['BatID'].isin(train_bats)].index.tolist()
    val_indices = df[df['BatID'].isin(val_bats)].index.tolist()
    test_indices = df[df['BatID'].isin(test_bats)].index.tolist()
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    # Print split information
    print("\nDataset Split Information:")
    print(f"Total number of bats: {len(unique_bats)}")
    print(f"Number of training bats: {len(train_bats)}")
    print(f"Number of validation bats: {len(val_bats)}")
    print(f"Number of test bats: {len(test_bats)}")
    print(f"\nTotal number of images: {len(df)}")
    print(f"Number of training images: {len(train_indices)}")
    print(f"Number of validation images: {len(val_indices)}")
    print(f"Number of test images: {len(test_indices)}")
    
    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Training loop
def train_model_enhanced(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs, 
                device,
                save_path='best_bifpn_model.pth'):
    """
    Training loop for U-Net with Bi-FPN model
    """
    # Initialize best validation loss
    best_val_loss = float('inf')
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    
    print(f"Starting training at {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Main epoch loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        
        print("Training...")
        for images, masks in tqdm(train_loader):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss = criterion(predictions, masks)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                print(f"Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        batch_count = 0
        
        print("\nValidating...")
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                predictions = model(images)
                
                # Calculate loss
                loss = criterion(predictions, masks)
                
                # Update metrics
                val_loss += loss.item()
                batch_count += 1
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch + 1}/{num_epochs} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f'Saved new best model with validation loss: {avg_val_loss:.4f}')
    
    # Final plot
    plot_training_history(train_losses, val_losses)
    
    return train_losses, val_losses

# Function to evaluate model performance with metrics from paper
def evaluate_model_comprehensive(model, test_loader, criterion, device, threshold=0.5):
    """
    Comprehensive evaluation with metrics used in the paper:
    - Sensitivity (SE)
    - Specificity (SP)
    - Accuracy (ACC)
    - AUC
    """
    model.eval()
    
    # Initialize metrics
    test_loss = 0.0
    all_preds = []
    all_masks = []
    
    # For calculating pixel-wise metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Make predictions
            predictions = model(images)
            
            # Calculate loss
            loss = criterion(predictions, masks)
            test_loss += loss.item()
            
            # Convert to binary predictions using threshold
            binary_preds = (predictions > threshold).float()
            
            # Collect for ROC curve calculation
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())
            
            # Update confusion matrix values
            for i in range(len(images)):
                pred = binary_preds[i].view(-1)
                mask = masks[i].view(-1)
                
                total_tp += (pred * mask).sum().item()
                total_fp += (pred * (1 - mask)).sum().item()
                total_fn += ((1 - pred) * mask).sum().item()
                total_tn += ((1 - pred) * (1 - mask)).sum().item()
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate metrics
    sensitivity = total_tp / (total_tp + total_fn + 1e-5)
    specificity = total_tn / (total_tn + total_fp + 1e-5)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-5)
    
    # Calculate AUC if sklearn is available
    try:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(np.array(all_masks) > 0.5, all_preds)
    except:
        auc_score = 0.0
        print("sklearn not available, AUC not calculated")
    
    # Print results
    print("\nTest Results:")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Sensitivity (SE): {sensitivity:.4f}")
    print(f"Specificity (SP): {specificity:.4f}")
    print(f"Accuracy (ACC): {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")
    
    # Create a results dictionary
    results = {
        'test_loss': avg_test_loss,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'auc': auc_score
    }
    
    return results

# Model initialization and training example
def setup_training(model, learning_rate=1e-4, loss_type='combined'):
    """
    Set up loss function and optimizer for training
    
    Args:
        model: The U-Net model
        learning_rate: Learning rate for the optimizer
        loss_type: One of 'bce', 'dice', or 'combined'
        
    Returns:
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
    """
    # Set up loss function
    if loss_type == 'bce':
        criterion = nn.BCELoss()
    elif loss_type == 'dice':
        criterion = DiceLoss()
    elif loss_type == 'combined':
        criterion = CombinedLoss(dice_weight=0.2, bce_weight=0.8)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    return criterion, optimizer, scheduler

def predict_and_visualize(model, test_loader, device, num_examples=4):
    """
    Make predictions on test data and visualize the results with overlaid masks
    
    Args:
        model: Trained U-Net model
        test_loader: DataLoader for test data
        device: Device to run inference on (cuda or cpu)
        num_examples: Number of examples to visualize
    """
    model.eval()  # Set model to evaluation mode
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
    
    # Get a batch of test data
    images, true_masks = next(iter(test_loader))
    
    # If we have fewer examples than requested, adjust
    num_examples = min(num_examples, len(images))
    
    with torch.no_grad():  # No need to track gradients
        for i in range(num_examples):
            # Get a single example
            image = images[i:i+1].to(device)
            true_mask = true_masks[i:i+1].to(device)
            
            # Make prediction
            pred_mask = model(image)
            
            # Move everything back to CPU for visualization
            image = image.cpu().squeeze(0)
            true_mask = true_mask.cpu().squeeze(0)
            pred_mask = pred_mask.cpu().squeeze(0)
            
            # Convert tensors to numpy arrays
            image_np = image.squeeze().numpy()
            true_mask_np = true_mask.squeeze().numpy()
            pred_mask_np = pred_mask.squeeze().numpy()
            
            # Create overlay image (predicted mask in red)
            overlay = np.zeros((*image_np.shape, 3))
            overlay[..., 0] = image_np  # Red channel
            overlay[..., 1] = image_np  # Green channel
            overlay[..., 2] = image_np  # Blue channel
            
            # Add mask in red with 50% opacity (adjust color and alpha as needed)
            overlay[..., 0] = np.maximum(image_np, pred_mask_np * 0.7)
            overlay[..., 1] = image_np * (1 - pred_mask_np * 0.5)
            overlay[..., 2] = image_np * (1 - pred_mask_np * 0.5)
            
            # Ensure overlay values are within range
            overlay = np.clip(overlay, 0, 1)
            
            # Display images
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(true_mask_np, cmap='gray')
            axes[i, 1].set_title('True Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask_np, cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (Pred Mask)')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with adaptive number of levels:

# Initialize improved model with automatic level limitation
# For 48x48 patches, we should only use 3-4 levels to avoid 1x1 feature maps
model = UNetBiFPN(
    in_channels=1,  # 1 for grayscale
    out_channels=1, # 1 for binary segmentation
    features=[32, 64, 128, 256, 512],  # Feature dimensions at each level
    max_levels=4  # Limit to 4 levels max for 48x48 input
)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Setup with combined loss
criterion, optimizer, scheduler = setup_training(
    model,
    learning_rate=0.001,
    loss_type='combined'  # Use combined loss as in the paper
)

# Create data loaders with enhanced preprocessing
train_loader, val_loader, test_loader = create_train_val_test_splits_enhanced(
    csv_file='Dataset/dataset.csv',
    image_dir='Dataset/Images',
    mask_dir='Dataset//Masks',
    test_size=0.2,
    batch_size=2
)

# Train model
train_losses, val_losses = train_model_enhanced(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=20,  # As in the paper, using fewer epochs than original U-Net
    device=device
)

# Evaluate model
results = evaluate_model_comprehensive(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device=device
)

print(f"Final results: SP={results['specificity']:.4f}, SE={results['sensitivity']:.4f}, "
      f"ACC={results['accuracy']:.4f}, AUC={results['auc']:.4f}")

predict_and_visualize(model, test_loader, device, num_examples=4)