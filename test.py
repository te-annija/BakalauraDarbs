import numpy as np
from dataset import Pascal3DDataset
from sklearn.metrics import accuracy_score
from config import *
from utils.helper import load_model

IOU_THRESHOLD = 0.5

def calculate_iou(box1, box2):
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    intersection_area = max(0, x2_intersect - x1_intersect) * max(
        0, y2_intersect - y1_intersect
    )

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def evaluate_localization(true_localization, pred_localization):
    iou_scores = []
    for true_box, pred_box in zip(true_localization, pred_localization):
        iou = calculate_iou(true_box, pred_box)
        iou_scores.append(iou)

    mean_iou = np.mean(iou_scores)
    print(f"Detection Mean IoU: {mean_iou:.4f}")

    return iou_scores

def evaluate_classification(true_classifications, pred_classifications):
    if len(pred_classifications.shape) > 1 and pred_classifications.shape[1] > 1:
        pred_classifications = np.argmax(pred_classifications, axis=1)

    if len(true_classifications.shape) > 1 and true_classifications.shape[1] > 1:
        true_classifications = np.argmax(true_classifications, axis=1)

    accuracy = accuracy_score(true_classifications, pred_classifications)
    print(f"classification Accuracy: {accuracy:.4f}")

    return true_classifications, pred_classifications

def evaluate_regression(true_sin_cos, pred_sin_cos):
    true_angles = np.degrees(np.arctan2(true_sin_cos[:, 0], true_sin_cos[:, 1]))
    pred_angles = np.degrees(np.arctan2(pred_sin_cos[:, 0], pred_sin_cos[:, 1]))

    true_angles = (true_angles + 360) % 360
    pred_angles = (pred_angles + 360) % 360

    angle_diffs = np.minimum(
        np.abs(true_angles - pred_angles), 360 - np.abs(true_angles - pred_angles)
    )

    median_angle_error = np.median(angle_diffs)

    angle_thresholds = [10, 20, 30]
    for threshold in angle_thresholds:
        accuracy = np.mean(angle_diffs <= threshold)
        print(f"Angle Accuracy@{threshold}°: {accuracy:.4f}")

    print(f"Median Angular Error: {median_angle_error:.2f}°")

    return angle_diffs

def evaluate_avp_classification(ious, true_angles, pred_angles):
    correct = 0
    total = len(ious)

    for iou, true_label, pred_angle in zip(ious, true_angles, pred_angles):
        if iou >= IOU_THRESHOLD and true_label == pred_angle:
            correct += 1
    avp = correct / total if total > 0 else 0
    print(f"Average Precision (AVP): {avp:.4f}")

    return avp

def evaluate_avp_regression(ious, angle_diffs):
    correct = 0
    total = len(ious)

    for iou, angle_diff in zip(ious, angle_diffs):
        if iou >= IOU_THRESHOLD and angle_diff <= 30:
            correct += 1

    avp = correct / total if total > 0 else 0
    print(f"Average Precision (AVP): {avp:.4f}")

    return avp

def test_model(model, task, test_dataset):
    test_ds = test_dataset.get_dataset(task=task)

    if task in [LOCALIZATION_TASK, ANGLE_CLASSIFICATION_TASK, ANGLE_REGRESSION_TASK]:
        true_values = []
        pred_values = []

        for _, y in test_ds:
            true_values.extend(y.numpy())

        true_values = np.array(true_values)
        pred_values = model.predict(test_ds)

        if task == LOCALIZATION_TASK:
            return evaluate_localization(true_values, pred_values)
        elif task == ANGLE_CLASSIFICATION_TASK:
            return evaluate_classification(true_values, pred_values)
        elif task == ANGLE_REGRESSION_TASK:
            return evaluate_regression(true_values, pred_values)
    elif task in [MTL_CLASSIFICATION, MTL_REGRESSION]:
        true_localization = []
        true_angles = []

        for _, target in test_ds:
            true_localization.extend(target[LOCALIZATION_TASK].numpy())
            if task == MTL_CLASSIFICATION:
                true_angles.extend(target[ANGLE_CLASSIFICATION_TASK].numpy())
            elif task == MTL_REGRESSION:
                true_angles.extend(target[ANGLE_REGRESSION_TASK].numpy())

        true_localization = np.array(true_localization)
        true_angles = np.array(true_angles)

        mtl_pred = model.predict(test_ds)

        pred_localization = mtl_pred[0]
        pred_angles = mtl_pred[1]

        ious = evaluate_localization(true_localization, pred_localization)
        if task == MTL_CLASSIFICATION:
            true_angles, pred_angles = evaluate_classification(true_angles, pred_angles)
            evaluate_avp_classification(ious, true_angles, pred_angles)
        elif task == MTL_REGRESSION:
            angle_diffs = evaluate_regression(true_angles, pred_angles)
            evaluate_avp_regression(ious, angle_diffs)


def main(task):
    test_dataset = Pascal3DDataset(mode="test", image_size=IMAGE_SIZE, shuffle=False)

    if task in TASKS:
        model = load_model(task)
        return test_model(model, task, test_dataset)

    model_localization = load_model(LOCALIZATION_TASK)
    ious = test_model(
        model=model_localization, task=LOCALIZATION_TASK, test_dataset=test_dataset
    )

    if task == "avp_classification":
        model_angles = load_model(ANGLE_CLASSIFICATION_TASK)
        true_angles, pred_angles = test_model(
            model=model_angles,
            task=ANGLE_CLASSIFICATION_TASK,
            test_dataset=test_dataset,
        )
        evaluate_avp_classification(ious, true_angles, pred_angles)
    elif task == "avp_regression":
        model_angles = load_model(ANGLE_REGRESSION_TASK)
        angle_diffs = test_model(
            model=model_angles, task=ANGLE_REGRESSION_TASK, test_dataset=test_dataset
        )
        evaluate_avp_regression(ious, angle_diffs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test EfficientNet model for Pascal3D+ tasks."
    )
    parser.add_argument("--task", default=MTL_REGRESSION, type=str)
    args = parser.parse_args()

    main(task=args.task)
