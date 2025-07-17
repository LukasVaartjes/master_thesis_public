import os
import json
import torch
from torch.utils.data import DataLoader
from dataset_pointcloud import PointCloudDataset
from pointnet_plus_plus import PointNetPlusPlusClassifier
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

results = []
MODEL_NAME = "pointcloud"
DATASET_DIR = "./dataset/"
SAVE_MODEL_PATH = "./dataset/saved_models"
SPLIT_OUTPUT_DIR = "split_output"
MODEL_PATH = f"{DATASET_DIR}/saved_models/{MODEL_NAME}/"
NUM_LABELS = 4
EXTRA_FEATURE = 0
NUM_POINTS = 2000

def classify_point_clouds(epoch):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PointCloudDataset(
        pointcloud_dir=f"{DATASET_DIR}/split_output/test",
        description_data=f"{DATASET_DIR}/split_output/test/test_labels.xlsx",
        num_points=NUM_POINTS
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(dataset)} samples for classification.")

    model = PointNetPlusPlusClassifier(num_classes=NUM_LABELS, extra_features_dim=EXTRA_FEATURE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    all_true_labels = []
    all_predicted_probs = []
    all_predicted_binary_labels = []

    counter = 1
    file_results_for_excel = []
    
    with torch.no_grad():
        for i, (points, extra_features, true_labels, pc_file) in enumerate(dataloader):

            if isinstance(pc_file, (tuple, list)):
                pc_file = pc_file[0]
            pc_file = str(pc_file)

            points = points.to(device)
            extra_features = extra_features.to(device)
            true_labels = true_labels.to(device)

            output = model(points, extra_features)

            probs = torch.sigmoid(output).cpu().detach().numpy()[0]
            
            initial_binary_predictions = (probs > 0.5).astype(int)
            
            final_pred_binary_labels = np.zeros_like(initial_binary_predictions, dtype=int)

            try:
                good_layer_idx = dataset.label_cols.index("Good_layer")
            except ValueError:
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

            file_results_for_excel.append({
                'Filename': pc_file,
                'True_Label': true_labels.cpu().numpy()[0].astype(int), 
                'Predicted_Label': pred_binary_labels, 
                'Predicted_Probability': probs 
            })
            
            true_label_str = ' '.join(f"{label}:{int(val)}" for label, val in zip(dataset.label_cols, true_labels.cpu().numpy()[0]))
            pred_label_str = ' '.join(f"{label}:{int(val)}" for label, val in zip(dataset.label_cols, pred_binary_labels))
            prob_str = ' '.join(f"{label}:{prob:.4f}" for label, prob in zip(dataset.label_cols, probs))

            is_correct_instance = np.array_equal(true_labels.cpu().numpy()[0], pred_binary_labels)
            color_start = "\033[92m" if is_correct_instance else "\033[91m"
            color_end = "\033[0m"

            print(f"Nr: {counter} {pc_file}: Probabilities -> {prob_str}")
            print(color_start + f"{pc_file}: \n Predicted: [{pred_label_str}], \n True       : [{true_label_str}]" + color_end)
            counter += 1
            
    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_predicted_binary_labels = np.array(all_predicted_binary_labels)

    per_label_f1 = []
    per_label_accuracy = []
    
    print("\nPer-Label Metrics:")
    for i, label_name in enumerate(dataset.label_cols):
        true_for_label = all_true_labels[:, i]
        pred_for_label = all_predicted_binary_labels[:, i]
        
        f1_lbl = f1_score(true_for_label, pred_for_label, zero_division=0)
        acc_lbl = np.mean(true_for_label == pred_for_label) * 100
        
        per_label_f1.append(f1_lbl)
        per_label_accuracy.append(acc_lbl)
        print(f"   {label_name}: Accuracy = {acc_lbl:.2f}%, F1 = {f1_lbl:.4f}")
        
        cm = confusion_matrix(true_for_label, pred_for_label)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['0', '1'],
                    yticklabels=['0', '1'])
        plt.title(f'Confusion Matrix for {label_name} (Epoch {epoch})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_output_dir = Path(f"{DATASET_DIR}/saved_models/{MODEL_NAME}/confusion_matrices/")
        cm_output_dir.mkdir(parents=True, exist_ok=True)
        cm_output_path = cm_output_dir / f"confusion_matrix_{label_name}_epoch_{epoch}.png"
        plt.savefig(cm_output_path)
        plt.close()
        print(f"Confusion matrix for {label_name} saved to {cm_output_path}")

    mean_label_accuracy = np.mean(per_label_accuracy)
    mean_label_f1 = np.mean(per_label_f1)
    print(f"\nMean Label Accuracy (avg per defect label): {mean_label_accuracy:.2f}%")
    print(f"Mean Label F1 Score (avg per defect label):: {mean_label_f1:.4f}")

    instance_accuracy = np.mean(np.all(all_true_labels == all_predicted_binary_labels, axis=1)) * 100
    print(f"Instance exact match Accuracy: {instance_accuracy:.2f}%")
    
    roc_data = {}
    for i, label_name in enumerate(dataset.label_cols):
        fpr_lbl, tpr_lbl, _ = roc_curve(all_true_labels[:, i], all_predicted_probs[:, i])
        roc_auc_lbl = auc(fpr_lbl, tpr_lbl)
        roc_data[label_name] = {'fpr': fpr_lbl, 'tpr': tpr_lbl, 'auc': roc_auc_lbl}

    output_dir = Path(f"{DATASET_DIR}/saved_models/{MODEL_NAME}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_output_path = output_dir / "multi_label_scores.txt"
    with open(txt_output_path, "a") as f:
        f.write(f"\n--- Epoch {epoch}, Model: {MODEL_NAME} ---\n")
        f.write(f"Mean Label Accuracy: {mean_label_accuracy:.2f}%\n")
        f.write(f"Mean Label F1 Score: {mean_label_f1:.4f}\n")
        f.write(f"Instance (Exact Match) Accuracy: {instance_accuracy:.2f}%\n")
        f.write("Per-Label Metrics:\n")
        for label_name, acc, f1 in zip(dataset.label_cols, per_label_accuracy, per_label_f1):
            f.write(f"    {label_name}: Accuracy = {acc:.2f}%, F1 = {f1:.4f}\n")
        f.write("ROC AUC for each label:\n")
        for label_name, data in roc_data.items():
            f.write(f"    {label_name}: AUC = {data['auc']:.4f}\n")

    print(f"Evaluation results saved to {txt_output_path}")

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

    for epoch in range(0, 151, 10):
        if not os.path.exists(MODEL_PATH):
            continue

        print(f"\nProcessing model at epoch {epoch}...")
        
        roc_data_for_epoch = classify_point_clouds(MODEL_PATH, epoch, MODEL_NAME)
        all_epochs_roc_data.append({'epoch': epoch, 'roc_data': roc_data_for_epoch})

    if all_epochs_roc_data:
        first_epoch_labels = list(all_epochs_roc_data[0]['roc_data'].keys())
    else:
        print("No ROC data generated for plotting")
        first_epoch_labels = []

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