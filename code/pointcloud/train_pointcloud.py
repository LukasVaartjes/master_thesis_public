# This script defines a  pipeline for classification model using PyTorch on pointcloud segments.
# sets up data loaders, defines a CNN model, implements training and validation loops,
# handles model checkpoints, and visualizes training progression.

import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch import nn
from dataset_pointcloud import PointCloudDataset
from pointnet_plus_plus import PointNetPlusPlusClassifier

# Constants and model settings
MODEL_NAME = "pointcloud"
DATASET_DIR = "./dataset/"
SAVE_MODEL_PATH = "./dataset/saved_models"
SPLIT_OUTPUT_DIR = "split_output"
TRAIN_DATA_DIR = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/train"
TRAIN_DATA_DESCRIPTION_FILE = f"{TRAIN_DATA_DIR}/train_labels.xlsx"
VAL_IMAGE_DIR = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/validate"
VAL_DESC = f"{DATASET_DIR}/{SPLIT_OUTPUT_DIR}/validate/validate_labels.xlsx"
EPOCHS = 13
BATCH_SIZE = 32
LR = 0.001
NUM_POINTS = 2000
# Number of output classes/labels
NUM_LABELS = 4
# Number of features used in the model now
EXTRA_FEATURES = 0
# Every % VAL_EPOCH validation run is done
VAL_EPOCH = 10


# Runs the entire training process for the image classification model
# 1. Sets up directories to save model and plots
# 2. Uses either gpu or cpu for training if available
# 3. Initializes the training and validation datasets using dataloaders
# 4. defines the Pointnet++ model with specified output labels and extra features.
# 5. Defines loss function and the optimizer
# 6. Sets up a learning rate scheduler to adjust the learning rate during training.
# 7. Implements the main training loop, iterating through epochs and batches.
# 8. Every 10 epochs validation run is performned and results are saved
# 9. Saves model checkpoints
# 10. Generates and saves a plot visualizing training loss, validation accuracy, and validation loss over epochs.
def train_pointcloud_model():
    save_dir = f"{SAVE_MODEL_PATH}/{MODEL_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    # Run on gpu if available, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Initialize datasetloader for training data
    train_dataset = PointCloudDataset(
        pointcloud_dir=TRAIN_DATA_DIR,
        description_data=TRAIN_DATA_DESCRIPTION_FILE,
        num_points=NUM_POINTS
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataloader for training data, nr of samples: {len(train_dataset)}, nr of batches: {len(train_dataloader)}")

    #Initialize datasetloader for validation set
    val_dataset = PointCloudDataset(
        pointcloud_dir=VAL_IMAGE_DIR,
        description_data=VAL_DESC,
        num_points=NUM_POINTS
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataloader for validation data, nr of samples: {len(val_dataset)}, nr of batches:{len(val_dataloader)}")

    #Initialize used model
    model = PointNetPlusPlusClassifier(num_classes=NUM_LABELS, extra_features_dim=EXTRA_FEATURES).to(device)
    print(f"Model architecture: {model}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    #learning rate where it is reduced by 0.5 every 10 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    
    def validate_model():
        """
        Performs a validation run over the validation dataset, in this case every 10 epochs
        Model is set to evaluation mode, iterates through dataitems from validation dataloader
        calculates predictions, validation loss and determines label and instance accuracy

        Returns:
            - float: The accuracy on validation set as a percentage
            - float: The loss on the validation set
        """
        model.eval()
        correct_predictions_per_label = torch.zeros(NUM_LABELS).to(device)
        total_predictions_per_label = torch.zeros(NUM_LABELS).to(device)
        all_labels_correct = 0
        total_samples = 0
        validation_loss = 0.0

       #for validation run dont use gradient calculations
        with torch.no_grad():
            for points, extra_features, labels, _ in val_dataloader:
                points, labels, extra_features = points.to(device), labels.to(device), extra_features.to(device)
                
                labels = labels.float()

                outputs = model(points, extra_features)
                
                # Calculate validation loss
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5

                # Calculate accuracy of predictions per label 
                correct_predictions_per_label += (preds == labels).sum(dim=0)
                total_predictions_per_label += labels.size(0)
                
                # Calculate accuracy of predictions per instance (all labels of single datapoint)
                all_labels_correct += (preds == labels).all(dim=1).sum().item()
                total_samples += labels.size(0)
                
        # Calculate average validation loss
        avg_validation_loss = validation_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0

        # Caluclate overal accuracy per label
        accuracy_per_label = torch.where(total_predictions_per_label > 0, 
                                         (correct_predictions_per_label / total_predictions_per_label) * 100, 
                                         torch.tensor(0.0).to(device))
        mean_accuracy = torch.mean(accuracy_per_label).item()

        # Calculate overall instance accuracy
        instance_accuracy = (all_labels_correct / total_samples * 100) if total_samples > 0 else 0.0

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

        for batch_idx, (points, extra_features, labels, filenames) in enumerate(progress_bar):
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

            # Debug purposes, incomment to remove
            label_names = ['Good_layer', 'Ditch', 'Crater', 'Waves'] # Define your label names
            probabilities = torch.sigmoid(outputs)
            # Convert probabilities to binary predictions (0 or 1 based on 0.5 threshold)
            preds_binary = (probabilities > 0.5).int()

            print(f"\n--- Epoch {epoch+1}, Batch {batch_idx+1} ---")
            for i in range(labels.size(0)): # Iterate through each sample in the batch
                prob_str = ', '.join(f'{label_names[j]}: {probabilities[i][j].item():.4f}' for j in range(len(label_names)))
                # Using .cpu().numpy().tolist() for labels to avoid ValueError if not single scalar
                print(f"File: {filenames[i]}, True: {labels[i].cpu().numpy().tolist()}, Pred: {preds_binary[i].cpu().numpy().tolist()}, Probs: {{{prob_str}}}")

        acc = (correct_total_labels / total_samples * 100) if total_samples > 0 else 0.0
        print_epoch_summary(epoch + 1, total_loss, acc)
        total_loss_array.append(total_loss)
        #Update learning rate scheduler
        scheduler.step()

        # Every 10 epochs, validate the model and save  model states
        if (epoch + 1) % VAL_EPOCH == 0:
            val_accuracy, val_loss = validate_model()
            val_acc_array.append(val_accuracy)
            val_loss_array.append(val_loss)
            print(f"validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")
            print(f"validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

            checkpoint_path = f"{save_dir}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    #plotting for Training Loss & Validation Accuracy
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(range(1, EPOCHS + 1), total_loss_array, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    
    val_epochs = [(i + 1) for i in range(EPOCHS) if (i + 1) %  VAL_EPOCH == 0]
    ax2.plot(val_epochs, val_acc_array, label='Validation Accuracy', color='tab:orange')
    ax2.set_ylabel('Validation Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Training Loss & Validation Accuracy")
    fig.tight_layout()
    plt.grid(True)
    save_path = f"{save_dir}/training_loss_and_val_accuracy.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

    #Plot validation loss
    fig2, ax3 = plt.subplots(figsize=(10, 6)) 
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Loss', color='tab:blue')
    ax3.plot(val_epochs, val_loss_array, label='Validation Loss', color='tab:blue')
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3.set_ylim(bottom=0) 
    ax3.legend(loc='upper right') 

    plt.title("Validation Loss")
    fig2.tight_layout()
    save_path_val_loss = f"{save_dir}/validation_loss.png"
    fig2.savefig(save_path_val_loss)
    plt.close(fig2)
    print(f"Plot saved to {save_path_val_loss}")

def print_epoch_summary(epoch, total_loss, accuracy):
    """
    Prints information about epoch training progress 

    Args:
        epoch (int): Epoch number
        total_loss (float): training loss for current epoch.
        accuracy (float):  accuracy for the current epoch.
    """
    green_color = "\033[32m"
    reset_color = "\033[0m"
    print(f"{green_color}[{epoch}/{EPOCHS}] Epoch completed. Total Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%{reset_color} \n")


if __name__ == "__main__":
    print(f"starting training for {MODEL_NAME}")
    train_pointcloud_model()