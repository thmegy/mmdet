import argparse
import mmdet.apis
import json
import os
import cv2 as cv
import tqdm
import numpy as np

import time



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, help="mmdetection checkpoint")
    parser.add_argument("--dataset", required=True, help="JSON cocolike dataset")
    parser.add_argument("--images-dir", required=True, help="Directory containing the images" )
    parser.add_argument("--viz-dir", required=True, help="Directory where visualizations will be saved")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="iou threshold")
    args = parser.parse_args()

    os.makedirs(args.viz_dir, exist_ok=True)
    
    return args


def iou(bbox1, bbox2):
    """ Compute Intersection over Union between the two bboxes.

    bbox1 and bbox2 are [x1, y1, x2, y2].
    """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = area_intersection / float(area1 + area2 - area_intersection)
    return iou


def draw_text(image, true_positives, false_positives, false_negatives):
    """ Draw colored informational text on the images. """
    # A shadow for the text so that we can read it no matter the background
    def draw_shadow(text, y):
        cv.putText(
            image,
            text,
            (image.shape[1] - 1000, y),
            cv.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )

    def draw_color(text, y, color):
        cv.putText(
            image,
            text,
            (image.shape[1] - 1000, y),
            cv.FONT_HERSHEY_SIMPLEX,
            3,
            color,
            3,
            cv.LINE_AA,
        )

    draw_shadow(f"True positives: {true_positives}", image.shape[0] - 300)
    draw_shadow(f"False positives: {false_positives}", image.shape[0] - 200)
    draw_shadow(f"False negatives: {false_negatives}", image.shape[0] - 100)
    draw_color(f"True positives: {true_positives}", image.shape[0] - 300, (0, 255, 0))
    draw_color(f"False positives: {false_positives}", image.shape[0] - 200, (0, 0, 255))
    draw_color(f"False negatives: {false_negatives}", image.shape[0] - 100, (255, 0, 0))


if __name__ == "__main__":
    args = parse_arguments()

    detector = mmdet.apis.init_detector(args.config, args.checkpoint, device="cuda:0")
    
    with open(args.dataset, "rt") as f_in:
        dataset = json.load(f_in)

    n_classes = len(dataset['categories'])

    bboxes_per_class = []
    for ic in range(n_classes):
        bboxes_per_image = {}
        for annot in dataset["annotations"]:
            if annot['category_id'] == ic:
                try:
                    bboxes_per_image[annot["image_id"]].append(annot["bbox"])
                except KeyError:
                    bboxes_per_image[annot["image_id"]] = []
                    bboxes_per_image[annot["image_id"]].append(annot["bbox"])
        bboxes_per_class.append(bboxes_per_image)

    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    for image_info in tqdm.tqdm(dataset["images"]):

        image_path = os.path.join(args.images_dir, image_info["file_name"])
        image = cv.imread(image_path)

        predictions = mmdet.apis.inference_detector(detector, image_path)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for ic, pred in enumerate(predictions): # loop on classes 
            try:
                gt_bboxes = bboxes_per_class[ic][image_info["id"]]
            except KeyError:
                # No bbox on that image
                gt_bboxes = []

            pred = [p for p in pred if p[4] > args.score_threshold]

            # Build iou matrix predictions vs gt
            iou_matrix = np.zeros((len(pred), len(gt_bboxes)))
            for i, p in enumerate(pred):
                for j, gt_bbox in enumerate(gt_bboxes):
                    x1_gt, y1_gt, width_gt, height_gt = gt_bbox
                    x2_gt = x1_gt + width_gt
                    y2_gt = y1_gt + height_gt
                    
                    bbox1 = x1_gt, y1_gt, x2_gt, y2_gt
                    bbox2 = p[:4]
                    iou_matrix[i, j] = iou(bbox1, bbox2)

            matches_matrix = iou_matrix > args.iou_threshold

            # Compute and draw results
            for i, gt_matches in enumerate(matches_matrix.T):
                num_matches = gt_matches.sum()
                if num_matches == 1:
                    # Draw true positives as green
                    pred_idx = gt_matches.argmax()
                    x1, y1, x2, y2 = pred[pred_idx][:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    true_positives += 1
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv.putText(image, f'{dataset["categories"][ic]["name"]}, {pred[pred_idx][4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv.LINE_AA)

                elif num_matches == 0:
                    false_negatives += 1
                    
                    # Draw false negatives as blue
                    x1, y1, width, height = gt_bboxes[i]
                    x2 = x1 + width
                    y2 = y1 + height
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv.putText(image, f'{dataset["categories"][ic]["name"]}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv.LINE_AA)

            for i, pred_matches in enumerate(matches_matrix):
                num_matches = pred_matches.sum()
                if num_matches == 0:                    
                    # Draw false negatives as red
                    x1, y1, x2, y2 = pred[i][:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    false_positives += 1
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv.putText(image, f'{dataset["categories"][ic]["name"]}, {pred[i][4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
                    
        draw_text(image, true_positives, false_positives, false_negatives)
        all_true_positives += true_positives
        all_false_positives += false_positives
        all_false_negatives += false_negatives

        viz_path = os.path.join(args.viz_dir, f"{image_info['file_name']}.jpg")
        cv.imwrite(viz_path, image)

#        print(viz_path, true_positives, false_positives, false_negatives)

    print(f"True positives: {all_true_positives}")
    print(f"False positives: {all_false_positives}")
    print(f"False negatives: {all_false_negatives}")
