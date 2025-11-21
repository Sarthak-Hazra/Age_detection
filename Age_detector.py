import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split 
import os
from torchvision import transforms
import scipy.io
from datetime import date, timedelta, datetime
import math

=

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = torch.sigmoid(avg_out + max_out)
        return x * channel_attention.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7),
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = torch.sigmoid(self.conv1(spatial_input))
        return x * spatial_attention.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, cbam=True):
        super(BasicBlock, self).__init__()
        self.cbam = cbam

        # Main Path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        if self.cbam:
            self.attn = CBAM(out_channels * self.expansion)

        # Shortcut (Identity or 1x1 Convolution)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.cbam:
            out = self.attn(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CBAMResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, img_channels=3):
        super(CBAMResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_channels, out_channels, stride_val, cbam=True))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out

def CBAM_ResNet18():
    return CBAMResNet(BasicBlock, [2, 2, 2, 2])


def datenum_to_datetime(datenum):
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

def calculate_age(photo_date_serial, dob_serial):
    try:
        age_in_days = photo_date_serial - dob_serial
        age = age_in_days / 365.25
        
        return float(age)
    except:
        return -1.0


def load_imdb_wiki_metadata(mat_file_path, image_dir):

    print(f"Loading metadata from: {mat_file_path}")
    
    try:
        mat = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {mat_file_path}. Check your path.")
        return [], []
    except Exception as e:
        print(f"ERROR loading .mat file: {e}")
        return [], []

    data = mat['wiki'][0, 0] if 'wiki' in mat else mat['imdb'][0, 0]
    
    # Extract raw data columns
    photo_taken = data[0][0] # Photo date in serial format
    full_path = data[2][0]   # Image path relative to the image_dir
    gender = data[3][0]
    date_of_birth = data[5][0] # DOB in serial format
    
    paths = []
    ages = []
    
    for i in range(len(full_path)):
        
        age_val = calculate_age(photo_taken[i], date_of_birth[i])
        
       
        rel_path = str(full_path[i][0])
        abs_path = os.path.join(image_dir, rel_path)

        
        if 1.0 <= age_val <= 100.0 and os.path.exists(abs_path):
            paths.append(abs_path)
            ages.append(age_val)

    return paths, ages

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, ages, transform=None):
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB') 
        except Exception as e:
            
            print(f"Could not load image {image_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor([30.0], dtype=torch.float32)

       
        if self.transform:
            image = self.transform(image)

        
        age = torch.tensor(self.ages[idx], dtype=torch.float32).unsqueeze(0) 
        
        return image, age

def get_data_loaders(mat_file_path, image_dir, batch_size=32, val_split=0.1):
    
    all_paths, all_ages = load_imdb_wiki_metadata(mat_file_path, image_dir)
    
    if not all_paths:
        print("ERROR: No valid data loaded. Check paths and dataset integrity.")
        return None, None


    train_paths, val_paths, train_ages, val_ages = train_test_split(
        all_paths, all_ages, test_size=val_split, random_state=42
    )

    print(f"Total valid samples: {len(all_paths)}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4), # Added common regularization technique
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(train_paths, train_ages, transform=train_transform)
    val_dataset = CustomImageDataset(val_paths, val_ages, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # Increased workers
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) # Increased workers

    return train_loader, val_loader


def calculate_mae(outputs, targets):
    return torch.mean(torch.abs(outputs - targets))

def validate_model(model, val_loader, device):
    """Evaluates the model on the validation set and returns the MAE."""
    model.eval() 
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad(): 
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            mae = calculate_mae(outputs, targets)
            total_mae += mae.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_mae = total_mae / total_samples
    model.train() 
    return avg_mae

def main():

    IMDB_WIKI_MAT_PATH = "./imdb_crop/imdb.mat" #put the path to the matlab file
    IMDB_WIKI_IMAGE_DIR = "./imdb_crop/" # put the path to the image files
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    epochs = 50 
    batch_size = 64
    learning_rate = 0.001


    model = CBAM_ResNet18().to(device)
    train_loader, val_loader = get_data_loaders(
        IMDB_WIKI_MAT_PATH, IMDB_WIKI_IMAGE_DIR, batch_size=batch_size
    )
    
    if train_loader is None or val_loader is None:
        return 

    print("Model initialized: CBAM-ResNet-18 for Age Regression")

    
    criterion = nn.L1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

   
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) 

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

       
        scheduler.step()

       
        val_mae = validate_model(model, val_loader, device)

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] | Train MAE (Loss): {avg_loss:.4f} | Val MAE: {val_mae:.4f}')

        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f'*** Saving best model with Val MAE: {best_val_mae:.4f} ***')
            torch.save(model.state_dict(), 'best_cbam_age_detector.pth')

    print('\nTraining complete.')

if __name__ == '__main__':
    main()