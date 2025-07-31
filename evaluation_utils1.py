import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve,auc

def calculate_iou(box1,box2):

    xA = max(box1[0],box2[0])
    yA = max(box1[1],box2[1])
    xB = min(box1[2],box2[2])
    yB = min(box1[3],box2[3])

    inter_area = max(0,xB - xA) * max(0,yB-yA)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area else 0
    return iou


def match_predictions(pred_boxes, gt_boxes, iou_threshold= 0.5):
    matches = []
    used_gt = set()

    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred['bbox'],gt['bbox'])
            if iou >= iou_threshold and iou > best_iou > best_iou and gt_idx not in used_gt:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >=0:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return matches, tp,fp,fn



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


# 4. AVERAGE PRECISION CALCULATION
def compute_ap(precisions, recalls):
    if len(precisions) < 2 or len(recalls) < 2:
        return 0.0
    return auc(recalls, precisions)


# 5. MEAN AVERAGE PRECISION (mAP)
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
    
import os

def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return [x1, y1, x2, y2]

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
                        entry['confidence'] = 0.0  # default value if confidence is missing
                        print(f"[Warning] Missing confidence score in {filename}, using 0.0")
                data.append(entry)
    return data


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


'''predictions = [
    {'class': 'Pan_Name', 'bbox': [100, 150, 200, 250], 'confidence': 0.92},
    {'class': 'Pan_Number', 'bbox': [50, 60, 120, 130], 'confidence': 0.88}
]

ground_truths = [
    {'class': 'Pan_Name', 'bbox': [98, 148, 202, 252]},
    {'class': 'Pan_Number', 'bbox': [48, 58, 122, 132]}
]

classes = ['Pan_Name', 'Pan_Number']'''


 # Paths to your YOLO txt files
pred_folder = '/home/mladmin/WorkSpace/old_data/Workspace/self/yolo3/yolov7/predictions/yolov7_results/labels'
gt_folder = '/home/mladmin/WorkSpace/old_data/Workspace/self/yolo3/ground-truth/labels'

check_class_ids(gt_folder)
check_class_ids(pred_folder)
# Define class list
classes = ["Aadhar_Name","Aadhar_Number", "Pan_Number", "Pan_Name", "Pan_Father_Name"]


# Load data
predictions = load_yolo_annotations(pred_folder, classes, with_conf=True)
ground_truths = load_yolo_annotations(gt_folder, classes, with_conf=False)

# Evaluate
ap_results, mean_ap = evaluate_map(predictions, ground_truths, classes, iou_threshold=0.5)
print("AP per class:", ap_results)
print("mAP:", mean_ap)

# 2. Calculate Precision, Recall, F1 per class
metrics = {}
for cls in classes:
    precisions, recalls, conf_list = evaluate_class(predictions, ground_truths, cls, iou_threshold=0.5)
    if len(precisions) == 0 or len(recalls) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = precisions[-1]
        recall = recalls[-1]
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
    metrics[cls] = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 3. Save everything to CSV
import csv

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





