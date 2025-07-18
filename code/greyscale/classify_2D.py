# Code to run classification for a pytorch based image classification model. 
# It can use single and multilabel data. 
# It automatically loads the data, models and automates the classification run
# Metrics such as F1-score, accuracy and ROC AUC are collected and plotted in 
# their corresponding graphs. Confusion matrixes are generated, and all predictions
# are saved in an excel sheet for easy reference later on.

import os
import torch
from torch.utils.data import DataLoader
from dataset_2D import ImageDataset
from cnn_model_2d import SimpleImageCNN
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,  f1_score, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Global variables
results = []
MODEL_NAME = "greyscale"
DATASET_DIR = "./dataset"
MODEL_PATH = "dataset\saved_models/greyscale/"
IMAGE_SIZE = (150,150)
NUM_LABELS = 4
EXTRA_FEATURE = 0

def classify_images(MODEL_PATH, epoch, MODEL_NAME):
    """
    Classifies images using a pre-trained model and evaluating the performance

    This function does:
    1. Sets up the test DataLoader
    2. Load the pre-trained model
    3. Iterates through the test dataset and do clasification 
    4. Applies a rule for "Good_layer" predictions to ensure exclusivity with other defect labels
    5. Collects true labels, predicted probabilities, and final binary predictions
    6. Calculates and prints per-label F1-scores and accuracy
    7. Generates confusion matrices
    8. Calculates and prints overall mean label accuracy, mean label F1 score,
       and instance-level accuracy.
    9. Computes ROC curve data 
    10. Saves metrics to text file
    11. Exports predictions true labels and probabilities to Excel file for alls amples

    Args:
        MODEL_PATH (str): path to  pre-trained model
        epoch (int): The epoch number of the loaded model
        MODEL_NAME (str): name of  model
    """
    IMAGE_DIR = f"{DATASET_DIR}/split_output/test"
    DESCRIPTION_FILE = f"{DATASET_DIR}/split_output/test/test_labels.xlsx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset for the test set
    # set batch size tp 1 for individual image processing
    dataset = ImageDataset(
        image_dir=IMAGE_DIR,
        description_data=DESCRIPTION_FILE,
        target_size=IMAGE_SIZE if IMAGE_SIZE else (128, 128)
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

     # Initialize model with the number of labels and extra features, then load the pre-trained weights
    model = SimpleImageCNN(num_labels=NUM_LABELS, extra_features_dim=EXTRA_FEATURE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Lists to store all true labels, predicted probabilities, and binary predictions
    all_true_labels = []
    all_predicted_probs = []
    all_predicted_binary_labels = []
    counter = 1
    file_results_for_excel = []
    
    # Disable gradient computation for inference to save memory
    with torch.no_grad():
        for i, (images, extra_features, true_labels, img_file) in enumerate(dataloader):

            if isinstance(img_file, (tuple, list)):
                img_file = img_file[0]
            img_file = str(img_file)

            images = images.to(device)
            extra_features = extra_features.to(device)
            true_labels = true_labels.to(device)    
            
            # Perform forward pass
            output = model(images, extra_features)
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(output).cpu().detach().numpy()[0]
            
            initial_binary_predictions = (probs > 0.5).astype(int)

            # Initialize binary prediction, will be adjusted by exclusive rule later on
            final_pred_binary_labels = np.zeros_like(initial_binary_predictions, dtype=int)

            # Apply exclusive rule that good_layer does not occur with other defects
            # If any specific defect is predicted, good layer is 0, otherwise it is 1
            # or is set to 1 as a fallback if nothing else was predicted.
            try:
                good_layer_idx = dataset.label_cols.index("Good_layer")
            except ValueError:
                print("good layer label not found in column, skip exclusive rule application.")
                final_pred_binary_labels = initial_binary_predictions
            else:
                any_specific_defect_predicted = False
                for j, label_name in enumerate(dataset.label_cols):
                    if j != good_layer_idx:
                        if initial_binary_predictions[j] == 1:
                            final_pred_binary_labels[j] = 1
                            any_specific_defect_predicted = True

                if any_specific_defect_predicted:
                    final_pred_binary_labels[good_layer_idx] = 0
                else:
                    if initial_binary_predictions[good_layer_idx] == 1:
                        final_pred_binary_labels[good_layer_idx] = 1
                    else:
                        final_pred_binary_labels[good_layer_idx] = 1 
                        
            pred_binary_labels = final_pred_binary_labels

            all_true_labels.append(true_labels.cpu().numpy()[0])
            all_predicted_probs.append(probs)
            all_predicted_binary_labels.append(pred_binary_labels)

            # Store results for Excel
            file_results_for_excel.append({
                'Filename': img_file,
                'True_Label': int(true_labels.cpu().numpy().item()),
                'Predicted_Label': pred_binary_labels, 
                'Predicted_Probability': probs 
            })
            
            # labels and probabilities console output
            true_label_str = ' '.join(f"{label}:{int(val)}" for label, val in zip(dataset.label_cols, true_labels.cpu().numpy()[0]))
            pred_label_str = ' '.join(f"{label}:{int(val)}" for label, val in zip(dataset.label_cols, pred_binary_labels))
            prob_str = ' '.join(f"{label}:{prob:.4f}" for label, prob in zip(dataset.label_cols, probs))

            is_correct_instance = np.array_equal(true_labels.cpu().numpy()[0], pred_binary_labels)
            color_start = "\033[92m" if is_correct_instance else "\033[91m"
            color_end = "\033[0m"

            print(f"Nr: {counter} {img_file}: Probabilities -> {prob_str}")
            print(color_start + f"{img_file}: \n Predicted: [{pred_label_str}], \n True      : [{true_label_str}]" + color_end)
            counter += 1
            
    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_predicted_binary_labels = np.array(all_predicted_binary_labels)

    per_label_f1 = []
    per_label_accuracy = []
    
    print("\nPer-Label Metrics:")
    # Calculate and print metrics for each individual label
    for i, label_name in enumerate(dataset.label_cols):
        true_for_label = all_true_labels[:, i]
        pred_for_label = all_predicted_binary_labels[:, i]
        
        # Calculate F1-score and accuracy
        f1_lbl = f1_score(true_for_label, pred_for_label, zero_division=0)
        acc_lbl = np.mean(true_for_label == pred_for_label) * 100
        
        per_label_f1.append(f1_lbl)
        per_label_accuracy.append(acc_lbl)
        print(f"   {label_name}: Accuracy = {acc_lbl:.2f}%, F1 = {f1_lbl:.4f}")
        
        # Generate and save confusion matrix for each label
        cm = confusion_matrix(true_for_label, pred_for_label)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['0', '1'],
                    yticklabels=['0', '1'])
        plt.title(f'Confusion Matrix for {label_name} (Epoch {epoch})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_output_path = f"{DATASET_DIR}/saved_models/confusion_matrix_{label_name}_epoch_{epoch}.png"
        plt.savefig(cm_output_path)
        plt.close()
        print(f"Confusion matrix for {label_name} saved to {cm_output_path}")

    # Calculate overall mean metrics
    mean_label_accuracy = np.mean(per_label_accuracy)
    mean_label_f1 = np.mean(per_label_f1)
    print(f"\nMean Label Accuracy (avg per defect label): {mean_label_accuracy:.2f}%")
    print(f"Mean Label F1 Score (avg per defect label):: {mean_label_f1:.4f}")

    # Calculate instance-level exact match accuracy
    instance_accuracy = np.mean(np.all(all_true_labels == all_predicted_binary_labels, axis=1)) * 100
    print(f"Instance exact match Accuracy: {instance_accuracy:.2f}%")
    
    roc_data = {}
    # Calculate ROC curve data for each label
    for i, label_name in enumerate(dataset.label_cols):
        fpr_lbl, tpr_lbl, _ = roc_curve(all_true_labels[:, i], all_predicted_probs[:, i])
        roc_auc_lbl = auc(fpr_lbl, tpr_lbl)
        roc_data[label_name] = {'fpr': fpr_lbl, 'tpr': tpr_lbl, 'auc': roc_auc_lbl}

    output_dir = Path(f"{DATASET_DIR}/saved_models/{MODEL_NAME}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
     # Save results to a text file
    txt_output_path = output_dir / "multi_label_scores.txt"
    with open(txt_output_path, "a") as f:
        f.write(f"\n--- Epoch {epoch}, Model: {MODEL_NAME} ---\n")
        f.write(f"Mean Label Accuracy: {mean_label_accuracy:.2f}%\n")
        f.write(f"Mean Label F1 Score: {mean_label_f1:.4f}\n")
        f.write(f"Instance (Exact Match) Accuracy: {instance_accuracy:.2f}%\n")
        f.write("Per-Label Metrics:\n")
        for label_name, acc, f1 in zip(dataset.label_cols, per_label_accuracy, per_label_f1):
            f.write(f"   {label_name}: Accuracy = {acc:.2f}%, F1 = {f1:.4f}\n")
        f.write("ROC AUC for each label:\n")
        for label_name, data in roc_data.items():
            f.write(f"   {label_name}: AUC = {data['auc']:.4f}\n")

    print(f"Evaluation results saved to {txt_output_path}")

    # Prepare data for Excel export
    df_results_data = []
    for item in file_results_for_excel:
        row = {'Filename': item['Filename']}
        row['True_Label'] = ','.join(map(str, item['True_Label'])) if isinstance(item['True_Label'], np.ndarray) else str(item['True_Label'])
        row['Predicted_Label'] = ','.join(map(str, item['Predicted_Label'])) if isinstance(item['Predicted_Label'], np.ndarray) else str(item['Predicted_Label'])
        row['Predicted_Probability'] = ','.join(map(lambda x: f"{x:.4f}", item['Predicted_Probability'])) if isinstance(item['Predicted_Probability'], np.ndarray) else f"{item['Predicted_Probability']:.4f}"
        df_results_data.append(row)

    df_results = pd.DataFrame(df_results_data)
    
    excel_output_path = output_dir / f"per_file_predictions_epoch_{epoch}.xlsx"
    
    writer = pd.ExcelWriter(excel_output_path, engine='xlsxwriter')

    df_results.to_excel(writer, sheet_name='Predictions', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Predictions']

    writer.close()
    print(f"Per-file predictions saved to {excel_output_path}")

    return roc_data


if __name__ == "__main__":
    all_epochs_roc_data = [] 

     # Iterate through saved model checkpoints, they are saved every 10 epochs
    for epoch in range(0, 151, 10):
        MODEL_PATH = f"dataset/saved_models/{MODEL_NAME}/model_epoch_{epoch}.pth"
        # if model does not exist, skip
        if not os.path.exists(MODEL_PATH):
            continue
        print(f"\nProcessing model at epoch {epoch}...")
        
        roc_data_for_epoch = classify_images(MODEL_PATH, epoch, MODEL_NAME)
        all_epochs_roc_data.append({'epoch': epoch, 'roc_data': roc_data_for_epoch})

    # Plot ROC curves for each label across all epochs
    if all_epochs_roc_data:
        first_epoch_labels = list(all_epochs_roc_data[0]['roc_data'].keys())
    else:
        print("No ROC data generated for plotting")
        first_epoch_labels = []
        
    # Generate a separate ROC plot for each label
    for label_name in first_epoch_labels:
        plt.figure(figsize=(10, 8))
        for epoch_data_item in all_epochs_roc_data:
            epoch = epoch_data_item['epoch']
            roc_data = epoch_data_item['roc_data']
            
            if label_name in roc_data:
                fpr = roc_data[label_name]['fpr']
                tpr = roc_data[label_name]['tpr']
                roc_auc = roc_data[label_name]['auc']
                plt.plot(fpr, tpr, lw=2, label=f"Epoch {epoch} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves Across Epochs for {label_name}')
        plt.legend(loc="lower right")
        plt.grid(True)

        output_dir = Path(f"{DATASET_DIR}/saved_models/{MODEL_NAME}/")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"combined_roc_curve_{label_name}.png"))
        plt.show()
        plt.close()