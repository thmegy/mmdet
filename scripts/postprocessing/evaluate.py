import argparse
import mmdet.apis
import mmengine
import mmcv
import json
import os
import cv2 as cv
import tqdm
import numpy as np


def get_iou(bbox1, bbox2):
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



def compute_average_precision(iou, score, threshold_iou):
    '''
    Compute precision for every true positive detection, and corresponding average precision.
    '''
    iou = np.array(iou)
    score = np.array(score)
    # sort examples
    sort_inds = np.argsort(-score)
    sort_iou = iou[sort_inds]
    
    # count true positive examples
    pos_inds = sort_iou > threshold_iou
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]
    
    # count not difficult examples
    pn_inds = sort_iou != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / pn
    ap = np.sum(precision) / total_pos

    return ap, precision[pos_inds], score[sort_inds][pos_inds], # AP, [precision_array], [score of true positives]


    
def main(args):
    detector = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=f'cuda:{args.gpu_id}'
    )
    config = mmengine.Config.fromfile(args.config)
    dataset_path = config.test_dataloader.dataset.data_root + config.test_dataloader.dataset.ann_file
    with open(dataset_path, "rt") as f_in:
        dataset = json.load(f_in)
        images_dir = config.test_dataloader.dataset.data_prefix['img']

    n_classes = len(dataset['categories'])

    # Get gt bboxes 
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

    # loop over images
    iou_list = [[] for _ in range(n_classes)]
    score_list = [[] for _ in range(n_classes)]
    for i, image_info in tqdm.tqdm(enumerate(dataset["images"])):
        if i==100:
            break
        image_path = os.path.join(images_dir, image_info["file_name"])
        preds = mmdet.apis.inference_detector(detector, image_path)

        bboxes = preds.pred_instances.numpy()['bboxes']
        scores = preds.pred_instances.numpy()['scores']
        labels = preds.pred_instances.numpy()['labels']

        for ic in range(n_classes): # loop on classes
            try:
                gt_bboxes = bboxes_per_class[ic][image_info["id"]]
            except KeyError:
                # No bbox on that image
                gt_bboxes = []

            mask_cls = labels == ic
            bboxes_cls = bboxes[mask_cls]
            scores_cls = scores[mask_cls]

            for bbox, score in zip(bboxes_cls, scores_cls):
                max_iou = 0
                for gt_bbox in gt_bboxes:
                    x1_gt, y1_gt, width_gt, height_gt = gt_bbox
                    x2_gt = x1_gt + width_gt
                    y2_gt = y1_gt + height_gt
                    
                    gt_bbox = x1_gt, y1_gt, x2_gt, y2_gt
                    iou_tmp = get_iou(gt_bbox, bbox)
                    if iou_tmp > max_iou:
                        max_iou=iou_tmp
                iou_list[ic].append(max_iou)
                score_list[ic].append(score)


    for ic in range(n_classes):
        ap, precision_list, scores_tp = compute_average_precision(iou_list[ic], score_list[ic], args.iou_threshold)
        print(ap)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, help="mmdetection checkpoint")
#    parser.add_argument("--viz-dir", required=True, help="Directory where visualizations will be saved")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()

    main(args)




### old code

#    os.makedirs(f'{args.viz_dir}/only_TP', exist_ok=True) 
#    os.makedirs(f'{args.viz_dir}/FN', exist_ok=True) 
#    os.makedirs(f'{args.viz_dir}/FP', exist_ok=True) 
#    os.makedirs(f'{args.viz_dir}/FP_FN', exist_ok=True) 
#
#        all_true_positives = 0
#    all_false_positives = 0
#    all_false_negatives = 0
#    for image_info in tqdm.tqdm(dataset["images"]):
#
#        image_path = os.path.join(images_dir, image_info["file_name"])
#        image = cv.imread(image_path)
#
#        predictions = mmdet.apis.inference_detector(detector, image_path)
#
#        true_positives = 0
#        false_positives = 0
#        false_negatives = 0
#
#        for ic, pred in enumerate(predictions): # loop on classes 
#            try:
#                gt_bboxes = bboxes_per_class[ic][image_info["id"]]
#            except KeyError:
#                # No bbox on that image
#                gt_bboxes = []
#                
#            pred = [p for p in pred if p[4] > args.score_threshold]
#
#            # Build iou matrix predictions vs gt
#            iou_matrix = np.zeros((len(pred), len(gt_bboxes)))
#            for i, p in enumerate(pred):
#                for j, gt_bbox in enumerate(gt_bboxes):
#                    x1_gt, y1_gt, width_gt, height_gt = gt_bbox
#                    x2_gt = x1_gt + width_gt
#                    y2_gt = y1_gt + height_gt
#                    
#                    bbox1 = x1_gt, y1_gt, x2_gt, y2_gt
#                    bbox2 = p[:4]
#                    iou_matrix[i, j] = iou(bbox1, bbox2)
#
#            matches_matrix = iou_matrix > args.iou_threshold
#
#            # Compute and draw results
#            for i, gt_matches in enumerate(matches_matrix.T):
#                num_matches = gt_matches.sum()
#                if num_matches >= 1:
#                    # Draw true positives as green
#                    pred_idx = gt_matches.argmax()
#                    x1, y1, x2, y2 = pred[pred_idx][:4]
#                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                    true_positives += 1
#                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#                    cv.putText(image, f'{dataset["categories"][ic]["name"]}, {pred[pred_idx][4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
#
#                elif num_matches == 0:
#                    false_negatives += 1
#                    
#                    # Draw false negatives as blue
#                    x1, y1, width, height = gt_bboxes[i]
#                    x2 = x1 + width
#                    y2 = y1 + height
#                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                    cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
#                    cv.putText(image, f'{dataset["categories"][ic]["name"]}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)
#
#            for i, pred_matches in enumerate(matches_matrix):
#                num_matches = pred_matches.sum()
#                if num_matches == 0:                    
#                    # Draw false negatives as red
#                    x1, y1, x2, y2 = pred[i][:4]
#                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                    false_positives += 1
#                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
#                    cv.putText(image, f'{dataset["categories"][ic]["name"]}, {pred[i][4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
#                    
#        draw_text(image, true_positives, false_positives, false_negatives)
#        all_true_positives += true_positives
#        all_false_positives += false_positives
#        all_false_negatives += false_negatives
#
#        if false_positives == 0 and false_negatives == 0:
#            outdir = 'only_TP'
#        elif false_positives == 0 and false_negatives > 0:
#            outdir = 'FN'
#        elif false_positives > 0 and false_negatives == 0:
#            outdir = 'FP'
#        else:
#            outdir = 'FP_FN'
#            
#        viz_path = f'{args.viz_dir}/{outdir}/{image_info["file_name"]}.jpg'
#        cv.imwrite(viz_path, image)
#
##        print(viz_path, true_positives, false_positives, false_negatives)
#
#    print(f"True positives: {all_true_positives}")
#    print(f"False positives: {all_false_positives}")
#    print(f"False negatives: {all_false_negatives}")
#
#def draw_text(image, true_positives, false_positives, false_negatives):
#    """ Draw colored informational text on the images. """
#    # A shadow for the text so that we can read it no matter the background
#    def draw_shadow(text, y):
#        cv.putText(
#            image,
#            text,
#            (image.shape[1] - 235, y),
#            cv.FONT_HERSHEY_SIMPLEX,
#            0.6,
#            (0, 0, 0),
#            2,
#            cv.LINE_AA,
#        )
#
#    def draw_color(text, y, color):
#        cv.putText(
#            image,
#            text,
#            (image.shape[1] - 235, y),
#            cv.FONT_HERSHEY_SIMPLEX,
#            0.6,
#            color,
#            1,
#            cv.LINE_AA,
#        )
#
#    draw_shadow(f"True positives: {true_positives}", image.shape[0] - 60)
#    draw_shadow(f"False positives: {false_positives}", image.shape[0] - 35)
#    draw_shadow(f"False negatives: {false_negatives}", image.shape[0] - 10)
#    draw_color(f"True positives: {true_positives}", image.shape[0] - 60, (0, 255, 0))
#    draw_color(f"False positives: {false_positives}", image.shape[0] - 35, (0, 0, 255))
#    draw_color(f"False negatives: {false_negatives}", image.shape[0] - 10, (255, 0, 0))
