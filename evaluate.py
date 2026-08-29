import os
import torch
import shap
import time
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision import transforms
from skimage.io import imread
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from collections import Counter
from collections import defaultdict

DIRS = []

def set_consts(dirs):
    global DIRS
    DIRS = dirs
    warnings.filterwarnings("ignore")


def analyze_dataset_similarity(image_root, bands, sample_size=50, bins=50):
    """
    Analyze similarity across multiple datasets by comparing pixel intensity distributions.

    Args:
        image_dirs (list of str): Paths to directories containing the images.
        bands (list of str): Names of the bands corresponding to the datasets.
        sample_size (int): Number of random images to sample from each dataset.
        bins (int): Number of bins for the histograms.
    """
    image_dirs = [os.path.join(image_root, x) for x in os.listdir(image_root)]
    
    print("analyzing datasets for similarity")
    if len(image_dirs) != len(bands):
        raise ValueError("Number of image directories must match the number of bands.")

    # Randomly sample a subset of images to reduce memory usage
    sampled_images = []
    for image_dir in image_dirs:
        image_files = sorted(os.listdir(image_dir))
        sampled_files = image_files[0:sample_size]
        sampled_images.append([os.path.join(image_dir, f) for f in sampled_files])

    # Collect intensity data for histograms
    pixel_intensity_data = {band: [] for band in bands}

    for band, band_images in zip(bands, sampled_images):        
        print(f"                                              Current band: {band}  ", end = "\r")
        idx = 0
        for img_path in band_images:
            idx += 1
            print(f"analyzing image #{idx}/{len(band_images)}", end = "\r")
            img = imread(img_path).flatten()  # Flatten for intensity distribution
            pixel_intensity_data[band].extend(img)

    # Plot histograms for comparison
    print("Starting graphing process                                                      ")
    bin_edges = np.linspace(0, 255, bins + 1)
    idx = 0
    histogram_results = {}

    for band, intensities in pixel_intensity_data.items():
        idx += 1
        print(f"Calculating histogram #{idx}/{len(pixel_intensity_data.items())}", end="\r")
        hist, _ = np.histogram(intensities, bins=bin_edges, density=True)
        histogram_results[band] = hist
    
    print("Finished calculating histograms")
    idx = 0
    for band, hist in histogram_results.items():
        idx += 1
        print(f"Plotting graph #{idx}/{len(histogram_results.items())}", end="\r")
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, hist, label=f"{band} (n={len(pixel_intensity_data[band])})", alpha=0.7)

    print("Finished visualization                                                           ")

    plt.title("Pixel Intensity Distribution Across Datasets")
    plt.xlabel("Pixel Intensity")               
    plt.ylabel("Frequency (Normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("Creating dataframe                                                                     ")
    stats_df = pd.DataFrame({
        "Band": bands,
        "Mean Intensity": [np.mean(pixel_intensity_data[band]) for band in bands],
        "Std Intensity": [np.std(pixel_intensity_data[band]) for band in bands],
        "Median Intensity": [np.median(pixel_intensity_data[band]) for band in bands]
    })
    print("Summary Statistics:\n", stats_df)

def prepare_flattened_data(dataset, base_image_dir):
    print("prepare_flattened_data()")
    X, y = [], []

    for idx in range(len(dataset)):
        print(f"Analyzing image #{idx + 1}/{len(dataset)}", end="\r")
        
        # Ensure image path is correctly joined
        img_path = os.path.join(base_image_dir, dataset.iloc[idx, 0])
        
        try:
            # Load image and flatten it
            img = np.asarray(Image.open(img_path)).flatten()
            label = dataset.iloc[idx, 5]  # Adjust label column as necessary
            X.append(img)
            y.append(label)
        except Exception as e:
            print(f"\nError processing image at {img_path}: {e}")

    return np.array(X), np.array(y)

def plot_precision_recall_curve(all_labels, all_scores, model_name):
    # Compute precision and recall across thresholds
    precision_recall_curve()
    precision, recall, _ = precision_recall_curve((all_labels == 1).astype(int), all_scores[:,1])
    average_precision = average_precision_score((all_labels == 1).astype(int), all_scores[:,1])

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

def plot_roc_curve(all_labels, all_scores, model_name):
    fpr, tpr, _ = roc_curve((all_labels == 1).astype(int), all_scores[:,1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve for "bottle" class (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['can', 'bottle', 'container'], yticklabels=['can', 'bottle', 'container'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_iou_distribution(iou_list, model_name):
    # Flatten all IoU values into a single list
    iou_values = [iou.flatten() for iou in iou_list if iou.size > 0]
    iou_values = np.concatenate(iou_values) if iou_values else np.array([])

    plt.figure(figsize=(8, 6))
    plt.hist(iou_values, bins=20, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title(f'IoU Distribution for {model_name}')
    plt.show()


def shap_analysis(model, dataloader, device, model_name):
    # Use SHAP for model interpretation
    explainer = shap.Explainer(model, dataloader)
    shap_values = explainer(dataloader)

    shap.summary_plot(shap_values, plot_type="bar", show=True)

def bounding_box_visualization(model, dataloader, device, model_name):
    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        
        # Forward pass
        with torch.no_grad():
            predictions = model(images)
        
        # Display the first image with predicted and true bounding boxes
        image = images[0].cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        
        for target, prediction in zip(targets, predictions):
            boxes_true = target['boxes'].cpu().numpy()
            boxes_pred = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box in boxes_true:
                plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='g', linewidth=2))
            for i, box in enumerate(boxes_pred):
                if scores[i] > 0.5:  # Only draw boxes with score above 0.5
                    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='r', linewidth=2))
        
        plt.title(f'Predicted vs Ground Truth Bounding Boxes ({model_name})')
        plt.show()

def vis_tpfpfn(image, tp_boxes, tp_labels, tp_scores, fp_boxes, fp_labels, fp_scores, fn_boxes, fn_labels):
    """
    Visualize the image with TP, FP, and FN bounding boxes.

    Parameters:
    - image: The image as a NumPy array (H, W, C).
    - tp_boxes, fp_boxes, fn_boxes: Bounding boxes for TP, FP, and FN as NumPy arrays (N, 4).
    - tp_labels, fp_labels, fn_labels: Labels for TP, FP, and FN as NumPy arrays (N,).
    - tp_scores, fp_scores: Prediction confidence scores for TP and FP as NumPy arrays (N,).
    """
    # Convert the image tensor to a NumPy array if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Plot False Negatives
    for box, label in zip(fn_boxes, fn_labels):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='blue', linestyle=':', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"FN: {label}", color='blue', fontsize=12, backgroundcolor='white')

    # Plot True Positives
    for box, label, score in zip(tp_boxes, tp_labels, tp_scores):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"TP: {label} ({score:.2f})", color='green', fontsize=12, backgroundcolor='white')

    # Plot False Positives
    for box, label, score in zip(fp_boxes, fp_labels, fp_scores):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='red', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"FP: {label} ({score:.2f})", color='red', fontsize=12, backgroundcolor='white')

    plt.axis('off')
    plt.show()

def visualize_predictions(image, boxes_true, labels_true, boxes_pred, labels_pred, scores_pred):
    """
    Visualize the image with ground truth and predicted bounding boxes.

    Parameters:
    - image: The image tensor (C, H, W) or (H, W, C), as a NumPy array.
    - boxes_true: Ground truth bounding boxes as a NumPy array (N, 4).
    - labels_true: Ground truth labels as a NumPy array (N,).
    - boxes_pred: Predicted bounding boxes as a NumPy array (M, 4).
    - labels_pred: Predicted labels as a NumPy array (M,).
    - scores_pred: Prediction confidence scores as a NumPy array (M,).
    """
    # Convert the image tensor to a NumPy array if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Plot ground truth boxes
    for box, label in zip(boxes_true, labels_true):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"GT: {label}", color='green', fontsize=12, backgroundcolor='white')

    # Plot predicted boxes
    i = 0
    for box, label, score in zip(boxes_pred, labels_pred, scores_pred):
        if i > 4:
            break
        i += 1
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"Pred: {label} ({score:.2f})", color='red', fontsize=12, backgroundcolor='white')

    plt.axis('off')
    plt.show()

def evaluate_model(model, model_name, csv_file, image_dir, device, iou_threshold=0.5, score_threshold=None, look=False, save_scores_file=None):
    """
    Evaluate the model using a CSV file containing ground truth data for bounding boxes and classes.

    Args:
        model: The object detection model to evaluate.
        csv_file: Path to the CSV file containing ground truth data with columns ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'].
        image_dir: Directory containing the images referenced in the CSV file.
        device: Device to run the model on ('cuda' or 'cpu').
        iou_threshold: IoU threshold to determine matches between predictions and ground truth.
        score_threshold: Minimum confidence score for a prediction to be considered.
        look: If True, visualize predictions during evaluation.
        save_scores_file: Path to a CSV file to save the confidence scores for statistical analysis.

    Returns:
        tp, fp, fn, precision, recall, accuracy, f1, all_labels, all_predictions, all_scores
    """

    # Initialize metrics
    tp, fp, fn = 0, 0, 0
    all_labels = []
    all_predictions = []
    all_scores = []
    if save_scores_file and os.path.exists(save_scores_file):
        with open(save_scores_file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            model_scores_data = [row for row in reader]  # Load existing rows as dictionaries
    if score_threshold is None:
        score_threshold = 0.25

    # Load ground truth data from CSV
    ground_truth = csv_file
    ground_truth_grouped = ground_truth.groupby('filename')

    # Map classes to numerical labels for consistency
    class_mapping = {'can': 1, 'bottle': 2, 'container': 3}
    ground_truth['class'] = ground_truth['class'].map(class_mapping)

    # Transform for preprocessing the image
    transform = transforms.Compose([transforms.ToTensor()])

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        index = 1
        for filename, group in ground_truth_grouped:
            print(f'Processing image {index} out of {len(ground_truth_grouped)}', end='\r')

            # Prepare ground truth boxes and labels for the current image
            true_boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values
            true_labels = group['class'].values  # Use the numerical labels

            # Load and preprocess the image
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Run the model on the image
            outputs = model(image_tensor)[0]

            # Extract predictions with confidence scores above the threshold
            pred_scores = outputs['scores'].cpu().numpy()
            valid_indices = pred_scores > score_threshold
            pred_boxes = outputs['boxes'][valid_indices].cpu().numpy()
            pred_labels = outputs['labels'][valid_indices].cpu().numpy()
            pred_scores = pred_scores[valid_indices]

            # Visualize predictions if enabled
            if look:
                visualize_predictions(
                    image=np.array(image),
                    boxes_true=true_boxes,
                    labels_true=true_labels,
                    boxes_pred=pred_boxes,
                    labels_pred=pred_labels,
                    scores_pred=pred_scores
                )

            # Calculate IoU matrix
            if pred_boxes.size > 0 and true_boxes.size > 0:
                iou_matrix = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes)).numpy()
            else:
                iou_matrix = np.zeros((len(pred_boxes), len(true_boxes)))

            # Match predictions with ground truth
            matched_gt = set()
            matched_pred = set()

            for pred_idx, pred_iou in enumerate(iou_matrix):
                max_iou = np.max(pred_iou)
                gt_idx = np.argmax(pred_iou)

                if max_iou >= iou_threshold and gt_idx not in matched_gt and pred_labels[pred_idx] == true_labels[gt_idx]:
                    # True Positive: Predicted box matches a ground truth box with the same class
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    all_labels.append(true_labels[gt_idx])
                    all_predictions.append(pred_labels[pred_idx])
                    all_scores.append(pred_scores[pred_idx])

            # Handle unmatched ground truth (false negatives)
            for gt_idx in range(len(true_boxes)):
                if gt_idx not in matched_gt:
                    fn += 1
                    all_labels.append(true_labels[gt_idx])
                    all_predictions.append(0)  # Placeholder for "no prediction"
                    all_scores.append(0.0)

            # Handle unmatched predictions (false positives)
            for pred_idx in range(len(pred_boxes)):
                if pred_idx not in matched_pred:
                    fp += 1
                    all_labels.append(0)  # Placeholder for "no ground truth"
                    all_predictions.append(pred_labels[pred_idx])
                    all_scores.append(pred_scores[pred_idx])
            
            # Save the confidence scores for later analysis
            if save_scores_file:
                for pred_idx, score in enumerate(pred_scores):
                    decision = 'T' if pred_idx in matched_pred else 'F'  # 'T' for true positive, 'F' for false positive
                    model_scores_data.append({
                        'modelname': model_name,
                        'filename': filename,
                        'confidence': score,
                        'decision': decision
                    })

            index += 1

    # Save the confidence scores to a CSV file for later analysis (if requested)
    if save_scores_file:
        scores_df = pd.DataFrame(model_scores_data)
        scores_df.to_csv(save_scores_file, index=False)
        print(f"Confidence scores saved to {save_scores_file}")

    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    effectiveness = (precision + recall + f1) / 3

    return [tp, fp, fn], [accuracy, precision, recall, f1, effectiveness], all_labels, all_predictions, all_scores

def do_evaluation(band, bs, test = None, path=None, t=None, wannasee = None):
    print(f"Evaluating Model_{bs}_{band}                                                     ")
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(DIRS[0], DIRS[2][0])
    save_scores_file = os.path.join(DIRS[0], 'zcode', 'confidences.csv')
    
    # Ensure the number of classes matches the training configuration
    csv_file_path = os.path.join(DIRS[0], DIRS[2][0], f"train_{band}.csv")
    test_data = pd.read_csv(csv_file_path)
    num_classes = test_data['class'].nunique() + 1  # Include background class
    model.head.classification_head.num_classes = num_classes

    if test:
        test_data = test_data[100:100+int(len(test_data)*test)]

    # Load the model's state_dict
    model_path = path if path else os.path.join(root, f'b{bs}_pth', f"model_{band}.pth")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_dir = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], f"chips_{band}")
    model_name = f"Model_{bs}_{band}"

    essentials, base_metrics, all_labels, all_predictions, all_scores = evaluate_model(model, model_name, test_data, image_dir, device, iou_threshold=0.5, score_threshold=t, look = wannasee, save_scores_file = save_scores_file)
    
    print(f"\nEvaluation Completed (TP {essentials[0]}, FP {essentials[1]}, FN {essentials[2]})")
    print(f"Accuracy: {base_metrics[0] * 100:.2f}%")
    print(f"Precision: {base_metrics[1]:.4f}")
    print(f"Recall: {base_metrics[2]:.4f}")
    print(f"F1-Score: {base_metrics[3]:.4f}")

    toWrite = [model_name] + essentials + base_metrics
    metrics_path = os.path.join(DIRS[0], DIRS[2][0], 'metrics.csv')
    metrics = pd.read_csv(metrics_path)
    new_row = pd.DataFrame([toWrite], columns = metrics.columns)
    metrics = pd.concat([metrics, new_row], ignore_index = True)
    metrics.to_csv(metrics_path, index=False)

    ## Confusion Matrix
    class_mapping = {'no object' : 0,
                     'can' : 1,
                     'bottle' : 2,
                     'container' : 3}
    valid_classes = [key for key in class_mapping]
    filtered_labels = [label for label in all_labels if label in class_mapping.values()]
    filtered_predictions = []
    for label in all_predictions:
        if label in class_mapping.values():
            filtered_predictions.append(label)
        else:
            filtered_predictions.append(class_mapping['no object'])
    '''
    print('i | true | pred')
    for i, (x,y) in enumerate(zip(filtered_labels, filtered_predictions)):
        print(f'{i} | {x} | {y}')
    '''
    counts = [0,0,0,0]
    for x in filtered_labels:
        counts[x] += 1
    print(counts)
    cm = confusion_matrix(filtered_labels, filtered_predictions, labels=range(len(valid_classes)))
    print('displaying matrix')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_classes)
    disp.plot(cmap="Blues")
    plt.show()

    """
    # Plot results
    try:
        cm = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(cm, model_name)
    except Exception as e:
        print(f"confusion failed: {e}")
    try:
        plot_roc_curve(all_labels, all_scores, model_name)
    except Exception as e:
        print(f"roc-plot failed: {e}")
    try:
        plot_precision_recall_curve(all_labels, all_scores, model_name)
    except Exception as e:
        print(f"prec-recall-plot failed: {e}")
    try:
        # SHAP analysis
        shap_analysis(model, eval_loader, device, model_name)
    except Exception as e:
        print(f"shap failed: {e}")
    # this is where the other set of three quotes was
    plot_iou_distribution(iou_list, model_name)    
    try:
        # Bounding box visualization
        bounding_box_visualization(model, dataloader, device, model_name)
    except Exception as e:
        print(f"bound-vis failed: {e}")
    """