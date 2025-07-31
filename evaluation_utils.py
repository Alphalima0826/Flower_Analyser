import numpy as np
from collections import defaultdict
from sklearn.metrics import auc
import os
import csv

# ---------- IOU ----------
def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area else 0
    return iou

# ---------- MATCH PREDICTIONS ----------
def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matches = []
    used_gt = set()

    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold and iou > best_iou and gt_idx not in used_gt:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return matches, tp, fp, fn

# ---------- EVALUATE CLASS ----------
def evaluate_class(preds, gts, class_name, iou_threshold=0.5):
    pred_filtered = [p for p in preds if p['class'] == class_name]
    gt_filtered = [g for g in gts if g['class'] == class_name]

    pred_filtered = sorted(pred_filtered, key=lambda x: x['confidence'], reverse=True)

    tp_list = []
    conf_list = []
    matched_gt = set()

    for pred in pred_filtered:
        iou_max = 0
        matched = False
        for gt_idx, gt in enumerate(gt_filtered):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold and iou > iou_max:
                iou_max = iou
                matched_idx = gt_idx
                matched = True
        if matched:
            tp_list.append(1)
            matched_gt.add(matched_idx)
        else:
            tp_list.append(0)
        conf_list.append(pred['confidence'])

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - x for x in tp_list])

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (len(gt_filtered) + 1e-6)

    return precisions, recalls, conf_list

# ---------- COMPUTE AP ----------
def compute_ap(precisions, recalls):
    if len(precisions) < 2 or len(recalls) < 2:
        return 0.0
    return auc(recalls, precisions)

# ---------- EVALUATE MAP ----------
def evaluate_map(preds, gts, class_names, iou_threshold=0.5):
    ap_per_class = {}
    for cls in class_names:
        precisions, recalls, _ = evaluate_class(preds, gts, cls, iou_threshold)
        if len(precisions) < 2 or len(recalls) < 2:
            print(f"[Warning] Not enough data points for AUC calculation in class '{cls}'")
        ap = compute_ap(precisions, recalls)
        ap_per_class[cls] = ap
    mAP = np.mean(list(ap_per_class.values()))
    return ap_per_class, mAP

# ---------- YOLO FORMAT TO BBOX ----------
def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return [x1, y1, x2, y2]

# ---------- LOAD YOLO ANNOTATIONS ----------
def load_yolo_annotations(folder_path, class_names, with_conf=False, img_w=640, img_h=640):
    data = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])

                # Skip invalid class IDs
                if class_id < 0 or class_id >= len(class_names):
                    print(f"[Warning] Skipping invalid class ID {class_id} in file: {filename}")
                    continue

                x_center, y_center, width, height = map(float, parts[1:5])
                bbox = yolo_to_bbox(x_center, y_center, width, height, img_w, img_h)
                entry = {
                    'class': class_names[class_id],
                    'bbox': bbox
                }
                if with_conf:
                    try:
                        entry['confidence'] = float(parts[5])
                    except IndexError:
                        entry['confidence'] = 0.0  # default confidence if missing
                        print(f"[Warning] Missing confidence score in {filename}, using 0.0")
                else:
                    entry['confidence'] = 1.0  # Default for ground truth
                data.append(entry)
    return data

# ---------- CHECK CLASS IDS IN A FOLDER ----------
def check_class_ids(folder):
    unique_ids = set()
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        cid = int(parts[0])
                        unique_ids.add(cid)
                    except:
                        pass
    print(f"[Check] Class IDs found in {folder}:", sorted(unique_ids))


# ---------- PRINT UNIQUE CLASS IDS (ALT FUNCTION) ----------
def print_unique_class_ids(folder, folder_name):
    unique_ids = set()
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    unique_ids.add(int(parts[0]))
    print(f"Unique class IDs in {folder_name}: {sorted(unique_ids)}")



# ------------- MAIN --------------

if __name__ == "__main__":

    # Define class list (your 5 classes)
    classes = ["Pan_Number", "Pan_Name", "Pan_Father_Name"]

    # Paths to YOLO txt files
    pred_folder = '/home/mladmin/WorkSpace/old_data/Workspace/self/yolo4/yolov7/predictions/yolov7_results/labels'
    gt_folder = '/home/mladmin/WorkSpace/old_data/Workspace/self/yolo4/ground-truth/labels'

    # Check class IDs first
    check_class_ids(gt_folder)
    check_class_ids(pred_folder)

    # Load data
    predictions = load_yolo_annotations(pred_folder, classes, with_conf=True)
    ground_truths = load_yolo_annotations(gt_folder, classes, with_conf=False)


    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(ground_truths)} ground truth boxes")

    # Evaluate mAP
    ap_results, mean_ap = evaluate_map(predictions, ground_truths, classes, iou_threshold=0.3)
    print("AP per class:", ap_results)
    print("mAP:", mean_ap)

    # Calculate Precision, Recall, F1 per class
    metrics = {}
    for cls in classes:
        try:
            precisions, recalls, _ = evaluate_class(predictions, ground_truths, cls, iou_threshold=0.5)
            if len(precisions) == 0 or len(recalls) == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                precision = precisions[-1]
                recall = recalls[-1]
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
        except Exception as e:
            print(f"[Error] while evaluating class {cls}: {e}")
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}

    print("\n[Class-wise Prediction and GT Distribution]")
    for cls in classes:
        pred_count = sum(1 for p in predictions if p['class'] == cls)
        gt_count = sum(1 for g in ground_truths if g['class'] == cls)
        print(f"{cls}: {pred_count} predictions, {gt_count} ground truth boxes")

    print_unique_class_ids(gt_folder, "Ground Truths")
    print_unique_class_ids(pred_folder, "Predictions")

    # Save report to CSV
    report_path = 'evaluation_report.csv'
    csv_header = ['Class', 'Precision', 'Recall', 'F1 Score', 'AP']

    with open(report_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for class_name in ap_results:
            precision = metrics[class_name]['precision']
            recall = metrics[class_name]['recall']
            f1 = metrics[class_name]['f1']
            ap = ap_results[class_name]

            writer.writerow([class_name, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{ap:.4f}"])

        writer.writerow(['mAP', '-', '-', '-', f"{mean_ap:.4f}"])

    print(f"CSV Report saved to {report_path}")

