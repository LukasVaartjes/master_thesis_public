import optuna
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from dataset_pointcloud import PointCloudDataset
from pointnet_plus_plus import PointNetPlusPlusClassifier

MODEL_NAME = "pointcloud_optimized"
DATASET_DIR = "./dataset_agreed/"
SAVE_MODEL_PATH = "./dataset_agreed/saved_models"
SPLIT_OUTPUT_DIR = "split_output"
TRAIN_DATA_DIR = f"{DATASET_DIR}{SPLIT_OUTPUT_DIR}/train"
TRAIN_DATA_DESCRIPTION_FILE = f"{TRAIN_DATA_DIR}/train_labels.xlsx"
VAL_DATA_DIR = f"{DATASET_DIR}{SPLIT_OUTPUT_DIR}/validate"
VAL_DESC = f"{VAL_DATA_DIR}/validate_labels.xlsx"

NUM_POINTS = 2048
NUM_LABELS = 4
EXTRA_FEATURES = 0
VAL_EPOCH = 10

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    
    # Load datasets
    train_dataset = PointCloudDataset(
        pointcloud_dir=TRAIN_DATA_DIR,
        description_data=TRAIN_DATA_DESCRIPTION_FILE,
        num_points=NUM_POINTS,
        target_per_class=300,
        train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = PointCloudDataset(
        pointcloud_dir=VAL_DATA_DIR,
        description_data=VAL_DESC,
        num_points=NUM_POINTS,
        train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PointNetPlusPlusClassifier(num_classes=NUM_LABELS, extra_features_dim=EXTRA_FEATURES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    EPOCHS_HPO = 30
    
    for epoch in range(EPOCHS_HPO):
        model.train()
        total_loss = 0.0
        for points, extra_features, labels, _, _ in train_loader:
            points, labels, extra_features = points.to(device), labels.to(device), extra_features.to(device)
            optimizer.zero_grad()
            outputs = model(points, extra_features)
            loss = criterion(outputs, labels.float())
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
        for points, extra_features, labels, _, _ in val_loader:
            points, labels, extra_features = points.to(device), labels.to(device), extra_features.to(device)
            outputs = model(points, extra_features)
            preds = (torch.sigmoid(outputs) > 0.5).cpu()
            all_labels.append(labels.cpu())
            all_preds.append(preds)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Trial {trial.number} completed, Validation F1: {val_f1:.4f}")
    
    return val_f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (macro-F1): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
