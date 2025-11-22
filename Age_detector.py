import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import re
from tqdm import tqdm
import math

# --- 1. CONFIGURATION ---

# IMPORTANT: Replace this with the exact path to your UTKFace folder.
# The script will look for image files directly within this directory.
UTKFACE_DIR = r"C:\Users\Sarthak Hazra\Documents\vscode\age_estimation\UTKFace"

# Training parameters
# OPTIMIZATION 2: Increased BATCH_SIZE for better GPU utilization
BATCH_SIZE = 128 
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.1 # 10% of data for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- 2. CBAM MODULE IMPLEMENTATION ---
# # The CBAM module is composed of a Channel Attention Module (CAM) and a Spatial Attention Module (SAM).

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    It compresses the spatial dimensions and uses MLP to learn the channel relationships.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP: W1(W0(x)) where W0 is the reduction layer
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    It aggregates channel information through max and avg pooling along the channel axis.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Concatenate Avg and Max pooled features (2 channels) and apply convolution
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        
        attention_map = self.conv1(x_concat)
        return self.sigmoid(attention_map)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies Channel Attention followed by Spatial Attention.
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. Channel Attention
        ca_weight = self.ca(x)
        x = x * ca_weight # Element-wise multiplication
        
        # 2. Spatial Attention
        sa_weight = self.sa(x)
        x = x * sa_weight # Element-wise multiplication
        
        return x

# --- 3. RESNET-18 WITH CBAM (MODIFIED BASIC BLOCK) ---

class BasicBlockCBAM(nn.Module):
    """
    Basic Block for ResNet-18, modified to include CBAM after the two convolutions.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # CBAM component
        self.cbam = CBAM(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Shortcut connection to match dimensions if needed
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply CBAM before the final ReLU and shortcut addition
        out = self.cbam(out)
        
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNet18_CBAM(nn.Module):
    """
    ResNet-18 model using the BasicBlockCBAM.
    The final layer is configured for age regression (1 output).
    """
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet18_CBAM, self).__init__()
        self.in_planes = 64

        # Initial stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global Average Pooling 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global Average Pooling 
        out = self.avg_pool(out) 
        
        out = out.view(out.size(0), -1) # Flatten (Batch, 512, 1, 1) to (Batch, 512)
        out = self.linear(out)
        return out

def ResNet18CBAM():
    """ Instantiates the ResNet-18 CBAM model. """
    return ResNet18_CBAM(BasicBlockCBAM, [2, 2, 2, 2])


# --- 4. UTKFACE CUSTOM DATASET ---

class UTKFaceDataset(Dataset):
    """
    Custom PyTorch Dataset for the UTKFace dataset.
    Extracts age label from the image filenames.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.file_pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+).jpg')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Handle corrupted file or read error (e.g., return a placeholder or skip)
            print(f"Skipping file {img_name} due to error: {e}")
            # Recursively call __getitem__ with a new index (less than ideal but works)
            # For a production system, you'd prune the list upfront.
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        # Extract age from filename: [age]_[gender]_[race]_[date&time].jpg
        match = self.file_pattern.match(img_name)
        if match:
            age = int(match.group(1))
        else:
            # If parsing fails, use a placeholder and warn
            print(f"Could not parse age from filename: {img_name}. Using age 0.")
            age = 0 
        
        # Convert age to a float tensor for regression
        age_label = torch.tensor([float(age)], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, age_label

# Define transformations: Resizing, converting to tensor, and normalization
data_transforms = transforms.Compose([
    # OPTIMIZATION 1: Reduced size from 128x128 to 96x96 to speed up computation
    transforms.Resize((96, 96)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 5. TRAINING AND VALIDATION FUNCTIONS ---

def train_model(model, train_loader, criterion, optimizer):
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels) # MAE (L1 Loss) is common for age regression
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate_model(model, val_loader, criterion):
    """Performs validation and returns Mean Absolute Error (MAE) and loss."""
    model.eval()
    running_loss = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate Mean Absolute Error (MAE)
            # Age prediction is rounded to the nearest integer
            abs_error = torch.abs(outputs - labels)
            total_mae += torch.sum(abs_error).item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_mae = total_mae / len(val_loader.dataset)
    return epoch_loss, epoch_mae


# --- 6. MAIN EXECUTION ---

def main():
    # 1. Load Data
    full_dataset = UTKFaceDataset(root_dir=UTKFACE_DIR, transform=data_transforms)
    
    if len(full_dataset) == 0:
        print(f"Error: No images found in the directory: {UTKFACE_DIR}")
        print("Please check the UTKFACE_DIR path and ensure the folder contains JPG image files.")
        return

    # Split dataset into training and validation sets
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=(DEVICE.type == 'cuda') # OPTIMIZATION 3: Pin memory for faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=(DEVICE.type == 'cuda') # OPTIMIZATION 3: Pin memory for faster GPU transfer
    )

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # 2. Initialize Model, Loss, and Optimizer
    model = ResNet18CBAM().to(DEVICE)
    
    # For age regression, L1Loss (MAE) is a good choice as it penalizes errors linearly.
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_mae = float('inf')

    # 3. Training Loop
    print("\nStarting training...\n")
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Train
        train_loss = train_model(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_mae = validate_model(model, val_loader, criterion)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f} | Validation MAE: {val_mae:.2f}")
        
        # Save the best model based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint_path = 'best_age_estimator_cbam.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved best model to {checkpoint_path} with MAE: {best_val_mae:.2f}")

    print("\nTraining complete.")
    print(f"Best Validation MAE achieved: {best_val_mae:.2f} years")

if __name__ == '__main__':
    main()