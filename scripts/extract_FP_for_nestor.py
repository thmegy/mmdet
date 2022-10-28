import argparse
import mmdet.apis
import mmcv
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
    parser.add_argument("--viz-dir", required=True, help="Directory where visualizations will be saved")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()
   
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



if __name__ == "__main__":
    args = parse_arguments()

    score_threshold = [0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.25, 0.3, 0.3, 0.5, 0.35, 0.5]
    
    detector = mmdet.apis.init_detector(args.config, args.checkpoint, device=f'cuda:{args.gpu_id}')
    config = mmcv.Config.fromfile(args.config)
    dataset_path = config.data.test.ann_file
    images_dir = config.data.test.img_prefix
    
    with open(dataset_path, "rt") as f_in:
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

        
    for image_info in tqdm.tqdm(dataset["images"]):

        image_path = os.path.join(images_dir, image_info["file_name"])
        image = cv.imread(image_path)

        predictions = mmdet.apis.inference_detector(detector, image_path)

        for ic, pred in enumerate(predictions): # loop on classes
            image_tmp = image.copy()
            class_name = dataset["categories"][ic]["name"]
            os.makedirs(f'{args.viz_dir}/{class_name}', exist_ok=True)
            
            try:
                gt_bboxes = bboxes_per_class[ic][image_info["id"]]
            except KeyError:
                # No bbox on that image
                gt_bboxes = []
                
            pred = [p for p in pred if p[4] > score_threshold[ic]]

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
            false_positives = 0
            for i, pred_matches in enumerate(matches_matrix):
                num_matches = pred_matches.sum()
                if num_matches == 0:                    
                    # Draw false negatives as red
                    x1, y1, x2, y2 = pred[i][:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    false_positives += 1
                    cv.rectangle(image_tmp, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv.putText(image_tmp, f'{class_name}, {pred[i][4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
                                
            if false_positives > 0:
                viz_path = f'{args.viz_dir}/{class_name}/{image_info["file_name"]}'
                cv.imwrite(viz_path, image_tmp)
                ann_file = image_path.replace(".jpg",".txt").replace("(","\\(").replace(")","\\)")
                os.system(f'cp {ann_file} {args.viz_dir}/{class_name}/')


