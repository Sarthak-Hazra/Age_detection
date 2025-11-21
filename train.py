import torch
import torch.nn as nn
from models.cbam_resnet import CBAM_ResNet18
from data.imdb_wiki_dataset import get_data_loaders 


#Have to update the path to a matlab file from imdb and the image directory too
IMDB_WIKI_MAT_PATH = "./imdb_crop/imdb.mat" 
IMDB_WIKI_IMAGE_DIR = "./imdb_crop/"         

EPOCHS = 50 
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'best_cbam_age_detector.pth'

def calculate_mae(outputs, targets):
    """Calculates Mean Absolute Error (MAE), the standard metric for age estimation."""
    return torch.mean(torch.abs(outputs - targets))

def validate_model(model, val_loader, device):
    """Evaluates the model on the validation set."""
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #Load Data
    train_loader, val_loader = get_data_loaders(
        IMDB_WIKI_MAT_PATH, IMDB_WIKI_IMAGE_DIR, batch_size=BATCH_SIZE
    )
    
    if train_loader is None or val_loader is None:
        return

  
    model = CBAM_ResNet18().to(device)
    print("Model initialized: CBAM-ResNet-18 for Age Regression")

    criterion = nn.L1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #Training Loop
    best_val_mae = float('inf')
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        
        # Training phase
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) 

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Validation phase
        val_mae = validate_model(model, val_loader, device)

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] | Train MAE (Loss): {avg_loss:.4f} | Val MAE: {val_mae:.4f}')

        # Save the best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f'*** Saving best model with Val MAE: {best_val_mae:.4f} ***')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print('\nTraining complete.')

if __name__ == '__main__':
    main()