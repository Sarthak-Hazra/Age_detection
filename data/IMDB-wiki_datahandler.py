import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import scipy.io
from datetime import date, timedelta
from sklearn.model_selection import train_test_split

#MATLAB Date

def datenum_to_date(datenum):
    # MATLAB serial date starts from day 1. 
    # The offset from Python's datetime(1, 1, 1) is 366 days.
    MATLAB_EPOCH = date(1, 1, 1) - timedelta(days=366)
    
    return MATLAB_EPOCH + timedelta(days=datenum)

def calculate_age(photo_date_serial, dob_serial):
    try:
        date_photo = datenum_to_date(photo_date_serial)
        date_dob = datenum_to_date(dob_serial)
        
        # Calculate difference in years
        age = date_photo.year - date_dob.year - (
            (date_photo.month, date_photo.day) < (date_dob.month, date_dob.day)
        )
        return float(age)
    except Exception as e:
        return -1.0 

def load_imdb_wiki_metadata(mat_file_path, image_dir):
    print(f"Loading metadata from: {mat_file_path}")
    
    try:
        mat = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {mat_file_path}. Please check your IMDB_WIKI_MAT_PATH.")
        return [], []

    # 'wiki' or 'imdb' is the top-level structure in the .mat file
    key = 'wiki' if 'wiki' in mat else 'imdb'
    if key not in mat:
        print("ERROR: Could not find 'imdb' or 'wiki' key in the .mat file.")
        return [], []

    data = mat[key][0, 0]
    
    # Indices for data fields in the .mat structure
    photo_taken = data[0][0] 
    full_path = data[2][0] 
    date_of_birth = data[5][0]
    
    paths = []
    ages = []
    
    for i in range(len(full_path)):
        try:
            photo_date_num = photo_taken[i].item() 
            dob_num = date_of_birth[i].item()

            age_val = calculate_age(photo_date_num, dob_num)
        
            rel_path = str(full_path[i][0])
            abs_path = os.path.join(image_dir, rel_path)

            # Data Cleaning: Filter age range (1-100) and ensure the file exists
            if 1.0 <= age_val <= 100.0 and os.path.exists(abs_path):
                paths.append(abs_path)
                ages.append(age_val)
        except IndexError:
            continue
        except Exception:
            continue

    return paths, ages

class IMDBWIKIDataset(Dataset):
    def __init__(self, image_paths, ages, transform=None):
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        age = torch.tensor(self.ages[idx], dtype=torch.float32).unsqueeze(0) 

        try:
            image = Image.open(image_path).convert('RGB') 
        except Exception:
        
            print(f"Warning: Failed to load image at {image_path}. Returning zero tensor.")
            return torch.zeros(3, 224, 224), age 

        if self.transform:
            image = self.transform(image)
        
        return image, age

def get_data_loaders(mat_file_path, image_dir, batch_size=32, val_split=0.1, img_size=224):
    all_paths, all_ages = load_imdb_wiki_metadata(mat_file_path, image_dir)
    
    if not all_paths:
        print("Fatal: No valid samples found after loading and filtering metadata.")
        return None, None

    train_paths, val_paths, train_ages, val_ages = train_test_split(
        all_paths, all_ages, test_size=val_split, random_state=42
    )

    print(f"Total valid samples: {len(all_paths)}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_ages)}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation transformation 
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = IMDBWIKIDataset(train_paths, train_ages, transform=train_transform)
    val_dataset = IMDBWIKIDataset(val_paths, val_ages, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader