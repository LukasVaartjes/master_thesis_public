import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch import nn

from dataset_pointcloud import PointCloudDataset
from pointnet_plus_plus import PointNetPlusPlusClassifier


MODEL_NAME = "pointcloud"
DATASET_DIR = "./dataset/"
SAVE_MODEL_PATH = "./dataset/saved_models"
SPLIT_OUTPUT_DIR = "split_output"
TRAIN_DATA_DIR = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/train"
TRAIN_DATA_DESCRIPTION_FILE = f"{TRAIN_DATA_DIR}/train_labels.xlsx"
VAL_IMAGE_DIR = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/validate"
VAL_DESC = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/validate/validate_labels.xlsx"
EPOCHS = 15
BATCH_SIZE = 32
LR = 0.001
NUM_POINTS = 2000
NUM_LABELS = 4
EXTRA_FEATURES = 0


def train_pointcloud_model():
    save_dir = f"{SAVE_MODEL_PATH}/{MODEL_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = PointCloudDataset(
        pointcloud_dir=TRAIN_DATA_DIR,
        description_data=TRAIN_DATA_DESCRIPTION_FILE,
        num_points=NUM_POINTS
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataloader for training data, nr of samples: {len(train_dataset)}, nr of batches: {len(train_dataloader)}")

    val_dataset = PointCloudDataset(
        pointcloud_dir=VAL_IMAGE_DIR,
        description_data=VAL_DESC,
        num_points=NUM_POINTS
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataloader for validation data, nr of samples: {len(val_dataset)}, nr of batches:{len(val_dataloader)}")

    model = PointNetPlusPlusClassifier(num_classes=NUM_LABELS, extra_features_dim=EXTRA_FEATURES).to(device)
    print(f"Model architecture: {model}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    def print_epoch_summary(epoch, total_loss, accuracy):
        green_color = "\033[32m"
        reset_color = "\033[0m"
        print(f"{green_color}[{epoch}/{EPOCHS}] Epoch completed. Total Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%{reset_color} \n")

    def validate_model():
        model.eval()
        correct_predictions_per_label = torch.zeros(NUM_LABELS).to(device)
        total_predictions_per_label = torch.zeros(NUM_LABELS).to(device)
        all_labels_correct = 0
        total_samples = 0
        validation_loss = 0.0

        with torch.no_grad():
            for points, extra_features, labels, _ in val_dataloader:
                points, labels, extra_features = points.to(device), labels.to(device), extra_features.to(device)
                
                labels = labels.float()

                outputs = model(points, extra_features)
                
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5

                correct_predictions_per_label += (preds == labels).sum(dim=0)
                total_predictions_per_label += labels.size(0)
                
                all_labels_correct += (preds == labels).all(dim=1).sum().item()
                total_samples += labels.size(0)
                
        avg_validation_loss = validation_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0

        accuracy_per_label = torch.where(total_predictions_per_label > 0, 
                                         (correct_predictions_per_label / total_predictions_per_label) * 100, 
                                         torch.tensor(0.0).to(device))
        mean_accuracy = torch.mean(accuracy_per_label).item()

        instance_accuracy = (all_labels_correct / total_samples * 100) if total_samples > 0 else 0.0

        print(f"Validation - Mean Label Accuracy: {mean_accuracy:.2f}% | Instance Accuracy: {instance_accuracy:.2f}% | Validation Loss: {avg_validation_loss:.4f}")
        return instance_accuracy, avg_validation_loss

    total_loss_array = []
    val_acc_array = []
    val_loss_array = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_total_labels = 0
        total_samples = 0

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Starting training:")

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for batch_idx, (points, extra_features, labels, _) in enumerate(progress_bar):
            points, labels, extra_features = points.to(device), labels.to(device), extra_features.to(device)
            optimizer.zero_grad()

            labels = labels.float()

            outputs = model(points, extra_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct_total_labels += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        acc = (correct_total_labels / total_samples * 100) if total_samples > 0 else 0.0
        print_epoch_summary(epoch + 1, total_loss, acc)
        total_loss_array.append(total_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            val_accuracy, val_loss = validate_model()
            val_acc_array.append(val_accuracy)
            val_loss_array.append(val_loss)
            print(f"validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")
            print(f"validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

            checkpoint_path = f"{save_dir}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(range(1, EPOCHS + 1), total_loss_array, label='Training Loss', color='tab:blue', linestyle='-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    val_epochs = [(i + 1) for i in range(EPOCHS) if (i + 1) % 10 == 0]
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:orange')
    ax2.plot(val_epochs, val_acc_array, label='Validation Accuracy', color='tab:orange', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 100)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) 
    ax3.set_ylabel('Validation Loss', color='tab:red')
    ax3.plot(val_epochs, val_loss_array, label='Validation Loss', color='tab:red', linestyle=':')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_ylim(bottom=0) 

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')

    plt.title("Training Loss, Validation Accuracy, and Validation Loss")
    fig.tight_layout()
    plt.grid(True)
    save_path = f"{save_dir}/training_loss_val_accuracy_and_val_loss.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    print(f"starting training for {MODEL_NAME}")
    train_pointcloud_model()