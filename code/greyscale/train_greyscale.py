# This script defines a  pipeline for classification model using PyTorch on greyscale segments.
# sets up data loaders, defines a CNN model, implements training and validation loops,
# handles model checkpoints, and visualizes training progression.

import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch import nn
from dataset_2D import ImageDataset
from cnn_model_2d import SimpleImageCNN

MODEL_NAME = "greyscale"
DATASET_DIR = "./dataset/"
SAVE_MODEL_PATH = "./dataset/saved_models"
SPLIT_OUTPUT_DIR = "split_output"
TRAIN_DATA_DIR = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/train"
TRAIN_DATA_DESCRIPTION_FILE = f"{TRAIN_DATA_DIR}/train_labels.xlsx"
VAL_IMAGE_DIR = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/validate"
VAL_DESC = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/validate/validate_labels.xlsx"
EPOCHS = 150
BATCH_SIZE = 32
LR = 0.001
IMAGE_SIZE = (150, 150)
NUM_LABELS = 1
EXTRA_FEATURES = 0


# RUns the entire training process for the image classification model
# 1. Sets up directories to save model and plots
# 2. Uses either gpu or cpu for training if available
# 3. Initializes the training and validation datasets using dataloaders
# 4. defines the SimpleImageCNN model with specified output labels and extra features.
# 5. Defines loss function and the optimizer
# 6. Sets up a learning rate scheduler to adjust the learning rate during training.
# 7. Implements the main training loop, iterating through epochs and batches.
# 8. Every 10 epochs validation run is performned and results are saved
# 9. Saves model checkpoints
# 10. Generates and saves a plot visualizing training loss, validation accuracy, and validation loss over epochs.
def train_image_model():
    save_dir = f"{SAVE_MODEL_PATH}/{MODEL_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    # Run on gpu if available, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ImageDataset(
        image_dir=TRAIN_DATA_DIR,
        description_data=TRAIN_DATA_DESCRIPTION_FILE,
        target_size=IMAGE_SIZE
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataloader for training data, nr of batches: {len(train_dataloader)}")

    val_dataset = ImageDataset(
        image_dir=VAL_IMAGE_DIR,
        description_data=VAL_DESC,
        target_size=IMAGE_SIZE
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataloader for validation data, nr of batches:{len(val_dataloader)}")

    # I dont use any featuress at the moment
    # num_labels = len(train_dataset.label_cols)
    # extra_features_dim = len(train_dataset.extra_features_cols)

    model = SimpleImageCNN(num_labels=NUM_LABELS, extra_features_dim=EXTRA_FEATURES).to(device)
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
            for images, extra_features, labels, _ in val_dataloader:
                images, labels, extra_features = images.to(device), labels.to(device), extra_features.to(device)
                outputs = model(images, extra_features)

                # Calculate validation loss
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5

                correct_predictions_per_label += (preds == labels).sum(dim=0)
                total_predictions_per_label += labels.size(0)

                all_labels_correct += (preds == labels).all(dim=1).sum().item()
                total_samples += labels.size(0)
        
        # Calculate average validation loss
        avg_validation_loss = validation_loss / len(val_dataloader)

        accuracy_per_label = (correct_predictions_per_label / total_predictions_per_label) * 100
        mean_accuracy = torch.mean(accuracy_per_label).item()

        instance_accuracy = all_labels_correct / total_samples * 100

        print(f"Validation - Mean Label Accuracy: {mean_accuracy:.2f}% | Instance Accuracy: {instance_accuracy:.2f}% | Validation Loss: {avg_validation_loss:.4f}")
        return instance_accuracy, avg_validation_loss

    #Save intermediate model states for plotting later on
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

        for batch_idx, (images, extra_features, labels, _) in enumerate(progress_bar):
            images, labels, extra_features = images.to(device), labels.to(device), extra_features.to(device)
            optimizer.zero_grad()

            outputs = model(images, extra_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct_total_labels += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        acc = correct_total_labels / total_samples * 100
        print_epoch_summary(epoch + 1, total_loss, acc)
        total_loss_array.append(total_loss)
        scheduler.step()

        # Every 10 epochs, validate the model and save  model states
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

    #Plot training loss, validation accuracy, and validation loss
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
    print(f"\nStarting training for: {MODEL_NAME}")
    train_image_model()