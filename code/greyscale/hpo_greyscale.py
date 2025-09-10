import optuna
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from dataset_2D import ImageDataset
from cnn_model_2d import SimpleImageCNN

MODEL_NAME = "pointcloud"
DATASET_DIR = "./dataset_agreed/"
SAVE_MODEL_PATH = "./dataset/saved_models"
SPLIT_OUTPUT_DIR = "split_output"
TRAIN_DATA_DIR = f"{DATASET_DIR}{SPLIT_OUTPUT_DIR}/train"
TRAIN_DATA_DESCRIPTION_FILE = f"{TRAIN_DATA_DIR}/train_labels.xlsx"
VAL_IMAGE_DIR = f"{DATASET_DIR}{SPLIT_OUTPUT_DIR}/validate"
VAL_DESC = f"{DATASET_DIR}{SPLIT_OUTPUT_DIR}/validate/validate_labels.xlsx"
BATCH_SIZE = 32
LR = 0.001
NUM_POINTS = 512
# Number of output classes/labels
NUM_LABELS = 4
# Number of features used in the model now
EXTRA_FEATURES = 0
# Every % VAL_EPOCH validation run is done
VAL_EPOCH = 10
IMAGE_SIZE= (150,150)

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Load datasets with batch size suggested by Optuna
    train_dataset = ImageDataset(
        image_dir=os.path.join(TRAIN_DATA_DIR, 'png'),
        description_data=TRAIN_DATA_DESCRIPTION_FILE,
        target_size=IMAGE_SIZE,
        target_per_class=400,
        train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ImageDataset(
        image_dir=os.path.join(VAL_IMAGE_DIR, 'png'),
        description_data=VAL_DESC,
        target_size=IMAGE_SIZE,
        train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with suggested dropout
    model = SimpleImageCNN(num_labels=NUM_LABELS, extra_features_dim=EXTRA_FEATURES).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Train for a small number of epochs (for HPO, you can reduce epochs to save time)
    EPOCHS_HPO = 20
    
    for epoch in range(EPOCHS_HPO):
        model.train()
        total_loss = 0.0
        for images, extra_features, labels, _, _ in train_loader:
            images, labels, extra_features = images.to(device), labels.to(device), extra_features.to(device)
            optimizer.zero_grad()
            outputs = model(images, extra_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Trial {trial.number}, Epoch {epoch+1}/{EPOCHS_HPO}, Avg Training Loss: {avg_loss:.4f}")

    
    # Validation
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, extra_features, labels, _, _ in val_loader:
            images, labels, extra_features = images.to(device), labels.to(device), extra_features.to(device)
            outputs = model(images, extra_features)
            preds = (torch.sigmoid(outputs) > 0.5).cpu()
            all_labels.append(labels.cpu())
            all_preds.append(preds)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Trial {trial.number}, Epoch {epoch+1}/{EPOCHS_HPO}, Avg Training Loss: {avg_loss:.4f}, Epoch Val F1: {epoch_f1:.4f}")
    
    
    return f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (macro-F1): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
