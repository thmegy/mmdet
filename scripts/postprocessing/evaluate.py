import argparse
import mmdet.apis
import mmengine
import mmcv
import json
import os
import cv2 as cv
import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import pycocotools.mask as maskutils



def rle2mask(rle):
    '''
    Convert run-length encoding (RLE) to binary mask.
    '''
    rows, cols = rle['size']
    try:
        rlePairs = np.cumsum(np.array(rle['counts'])[:-1]).reshape(-1,2)
    except:
        rlePairs = np.cumsum(np.array(rle['counts'])).reshape(-1,2)
    
    mask = np.zeros(rows*cols,dtype=np.uint8)
    for start, end in rlePairs:
        mask[start-1:end] = 255
    mask = mask.reshape(cols, rows)
    return mask.T



def poly2mask(poly, height, width):
    '''
    Convert polygon to binary mask.
    height (width): height (width) of the image the annotation belongs to.
    '''
    rle = maskutils.merge(maskutils.frPyObjects(poly, height, width))
    mask = maskutils.decode(rle)
    return mask



def get_mask_iou(mask1, mask2):
    """ 
    Compute Intersection over Union between the two binary masks.
    Use sklearn implementation of Jaccard similarity score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score 
    """
    return jaccard_score(mask1, mask2, average='micro')



def get_bbox_iou(bbox1, bbox2):
    """ 
    Compute Intersection over Union between the two bboxes.
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



def compute_average_recall(scores_list, score_thrs):
    '''
    Compute recall for every threshold in <score_thrs>, and corresponding average recall.
    Inputs:
    scores_list: list of list of pred-bboxes scores matched to gt bboxes (N_gt_bboxes, n_matched)
    '''
    if (len(scores_list) == 0) or (len(score_thrs) == 0):
        return np.nan, []
    
    # loop in score thresholds, at each iteration compute number of FN
    recall_list = []
    for score_thr in score_thrs:
        #loop over gt bboxes
        tp_count = 0
        fp_count = 0
        for gt_matched_scores in scores_list:
            if (np.array(gt_matched_scores) >= score_thr).sum() > 0:
                tp_count +=1
            else:
                fp_count += 1
        recall = tp_count / (tp_count+fp_count) # TP / (TP+FN)
        recall_list.append(recall)

    ar = sum(recall_list) / len(recall_list)

    return ar, recall_list



def plot_precision_recall(results, outpath):
    figpr, axpr = plt.subplots() # precision-recall summary plot
    axpr.set_xlabel('recall')
    axpr.set_ylabel('precision')

    figf1, axf1 = plt.subplots(figsize=(8,6)) # F1-score summary plot
    axf1.set_xlabel('score threshold')
    axf1.set_ylabel('F1-score')

    max_f1_dict = {}
    max_sthr_dict = {}
    
    for cls, cls_dict in results.items():
        sthr_list = np.array(cls_dict['score_thrs'])
        precision_list = np.array(cls_dict['precision'])
        recall_list = np.array(cls_dict['recall'])
        f1_list = 2 * precision_list * recall_list / (precision_list+recall_list)

        axpr.plot(recall_list, precision_list, label=cls, marker='o', markersize=4)
        p = axf1.plot(sthr_list, f1_list, label=cls)
        col = p[-1].get_color()

        # get max f1 score and add to plot
        if len(f1_list) > 0:
            maxf1_id = np.argmax(f1_list)
            xmax = sthr_list[maxf1_id]
            ymax = f1_list[maxf1_id]
            axf1.plot([xmax, xmax], [0, ymax], color=col, linestyle='--', linewidth=1)
            axf1.plot([0, xmax], [ymax, ymax], color=col, linestyle='--', linewidth=1)
            plt.text(xmax, 0, f'{xmax:.2f}', color=col, horizontalalignment='right', verticalalignment='top', rotation=45, fontsize='small')
            plt.text(0, ymax, f'{ymax:.2f}', color=col, horizontalalignment='right', verticalalignment='center', fontsize='small')
        else:
            xmax, ymax = np.nan, np.nan
        results[cls]['optimal F1'] = ymax
        results[cls]['optimal score threshold'] = xmax

    axpr.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize='small')
    figpr.set_tight_layout(True)
    figpr.savefig(f'{outpath}/precision_recall.png')

    axf1.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize='medium')
    axf1.set_xlim(0)
    axf1.set_ylim(0)
    figf1.set_tight_layout(True)
    figf1.savefig(f'{outpath}/f1_score.png')

    plt.close('all')

    return results


    
def draw_text(image, true_positives, false_positives, false_negatives):
    """ Draw colored informational text on the images. """
    # A shadow for the text so that we can read it no matter the background
    def draw_shadow(text, y):
        cv.putText(
            image,
            text,
            (image.shape[1] - 235, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )

    def draw_color(text, y, color):
        cv.putText(
            image,
            text,
            (image.shape[1] - 235, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv.LINE_AA,
        )

    draw_shadow(f"True positives: {true_positives}", image.shape[0] - 60)
    draw_shadow(f"False positives: {false_positives}", image.shape[0] - 35)
    draw_shadow(f"False negatives: {false_negatives}", image.shape[0] - 10)
    draw_color(f"True positives: {true_positives}", image.shape[0] - 60, (0, 255, 0))
    draw_color(f"False positives: {false_positives}", image.shape[0] - 35, (0, 0, 255))
    draw_color(f"False negatives: {false_negatives}", image.shape[0] - 10, (255, 0, 0))



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

        
    # check if task is detection or instance segmentation
    segmentation = False
    if 'segmentation' in dataset["annotations"][0].keys():
        segmentation = True

    # get ground truth
    objects_per_class = [] # masks or bboxes, depending on task
    if segmentation:
        ### Get ground truth masks
        for cat_dict in dataset['categories']:
            cat_id = cat_dict['id']
            masks_per_image = {}
            for annot in dataset["annotations"]:
                if annot['category_id'] == cat_id:

                    ## identify format of segmentation (RLE or polygon) and convert to mask
                    if type(annot['segmentation']) == dict:
                        mask = rle2mask(annot['segmentation'])
                    else:
                        for im in dataset['images']:
                            if im['id'] == annot['image_id']:
                                image_ann = im
                        height = image_ann['height']
                        width = image_ann['width']
                        mask = poly2mask(annot['segmentation'], height, width)
                        
                    try:
                        masks_per_image[annot["image_id"]].append(mask)
                    except KeyError:
                        masks_per_image[annot["image_id"]] = []
                        masks_per_image[annot["image_id"]].append(mask)
            objects_per_class.append(masks_per_image)
    else:
        ### Get ground truth bboxes 
        for cat_dict in dataset['categories']:
            cat_id = cat_dict['id']
            bboxes_per_image = {}
            for annot in dataset["annotations"]:
                if annot['category_id'] == cat_id:
                    try:
                        bboxes_per_image[annot["image_id"]].append(annot["bbox"])
                    except KeyError:
                        bboxes_per_image[annot["image_id"]] = []
                        bboxes_per_image[annot["image_id"]].append(annot["bbox"])
            objects_per_class.append(bboxes_per_image)


    ### loop over images
    iou_list = [[] for _ in range(len(dataset['categories']))]
    score_list = [[] for _ in range(len(dataset['categories']))]
    gt_matched_scores = [[] for _ in range(len(dataset['categories']))] # list of list containing scores of predicted objects (bboxes or masks) with iou > thr with each gt bbox, for every classes
    inference_results = {'objects':[], 'scores':[], 'labels':[]}
    
    for image_info in tqdm.tqdm(dataset["images"]):
        image_path = os.path.join(images_dir, image_info["file_name"])
        preds = mmdet.apis.inference_detector(detector, image_path) ## inference

        if segmentation:
            objects = preds.pred_instances.numpy()['masks']
        else:
            objects = preds.pred_instances.numpy()['bboxes']
        scores = preds.pred_instances.numpy()['scores']
        labels = preds.pred_instances.numpy()['labels']

        # keep only objects with score > args.pre_score_thr, in order to reduce number of objects to process
        mask_score_pre = scores > args.pre_score_thr

        objects = objects[mask_score_pre]
        scores = scores[mask_score_pre]
        labels = labels[mask_score_pre]
        
        inference_results['objects'].append(objects)
        inference_results['scores'].append(scores)
        inference_results['labels'].append(labels)

        for ic in range(len(dataset['categories'])): # loop on classes
            try:
                gt_objects = objects_per_class[ic][image_info["id"]]
            except KeyError:
                # No object on that image
                gt_objects = []

            mask_cls = labels == ic
            objects_cls = objects[mask_cls]
            scores_cls = scores[mask_cls]
                
            gt_matched_scores_cls = [[] for _ in range(len(gt_objects))] # list of list containing scores of predicted objects with iou > thr with each gt bbox / mask
            for obj, score in zip(objects_cls, scores_cls):
                max_iou = 0
                for igt, gt_object in enumerate(gt_objects):
                    if segmentation:                        
                        iou_tmp = get_mask_iou(gt_object, obj)
                    else:
                        x1_gt, y1_gt, width_gt, height_gt = gt_object
                        x2_gt = x1_gt + width_gt
                        y2_gt = y1_gt + height_gt
                        gt_object = x1_gt, y1_gt, x2_gt, y2_gt

                        iou_tmp = get_bbox_iou(gt_object, obj)
                        
                    if iou_tmp > max_iou:
                        max_iou = iou_tmp
                    if iou_tmp > args.iou_threshold:
                        gt_matched_scores_cls[igt].append(score)
                iou_list[ic].append(max_iou)
                score_list[ic].append(score)
                
            gt_matched_scores[ic] += gt_matched_scores_cls


    ### compute AP, AR, and plot and save results
    results = {}
    for ic, cat_dict in enumerate(dataset['categories']):
        cls = cat_dict['name']
        if len(iou_list[ic]) > 0:
            results[cls] = {}
            ap, precision_list, scores_tp = compute_average_precision(iou_list[ic], score_list[ic], args.iou_threshold)
            ar, recall_list = compute_average_recall(gt_matched_scores[ic], scores_tp)

            results[cls]['ap'] = ap
            results[cls]['ar'] = ar
            results[cls]['score_thrs'] = scores_tp.tolist()
            results[cls]['precision'] = precision_list.tolist()
            results[cls]['recall'] = recall_list

    results = plot_precision_recall(results, args.outpath)

    with open(f'{args.outpath}/results.json', 'w') as fout:
        json.dump(results, fout, indent = 6)


    ### following works for bboxes only

    if not segmentation:

    
        ### Use optimised score thresholds to determine and draw True Positives, False Positives and False Negatives on the test images

        os.makedirs(f'{args.outpath}/test_images_with_prediction/only_TP', exist_ok=True) 
        os.makedirs(f'{args.outpath}/test_images_with_prediction/FN', exist_ok=True) 
        os.makedirs(f'{args.outpath}/test_images_with_prediction/FP', exist_ok=True) 
        os.makedirs(f'{args.outpath}/test_images_with_prediction/FP_FN', exist_ok=True) 

        all_true_positives = 0
        all_false_positives = 0
        all_false_negatives = 0
        for i, image_info in enumerate(tqdm.tqdm(dataset["images"])):
            image_path = os.path.join(images_dir, image_info["file_name"])
            image = cv.imread(image_path)

            bboxes = inference_results['objects'][i]
            scores = inference_results['scores'][i]
            labels = inference_results['labels'][i]
            
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for ic, cls in enumerate(results.keys()): # loop on classes
                try:
                    gt_bboxes = objects_per_class[ic][image_info["id"]]
                except KeyError:
                    # No bbox on that image
                    gt_bboxes = []

                score_thr = results[cls]['optimal score threshold']
                mask_score = scores > score_thr
                mask_cls = labels == ic
                bboxes_cls = bboxes[mask_cls & mask_score]
                scores_cls = scores[mask_cls & mask_score]

                # Build iou matrix predictions vs gt
                iou_matrix = np.zeros((len(bboxes_cls), len(gt_bboxes)))
                for i, bbox in enumerate(bboxes_cls):
                    for j, gt_bbox in enumerate(gt_bboxes):
                        x1_gt, y1_gt, width_gt, height_gt = gt_bbox
                        x2_gt = x1_gt + width_gt
                        y2_gt = y1_gt + height_gt

                        gt_bbox = x1_gt, y1_gt, x2_gt, y2_gt
                        iou_matrix[i, j] = get_bbox_iou(gt_bbox, bbox)

                matches_matrix = iou_matrix > args.iou_threshold

                # Compute and draw results
                for i, gt_matches in enumerate(matches_matrix.T):
                    num_matches = gt_matches.sum()
                    if num_matches >= 1:
                        # Draw true positives as green
                        pred_idx = gt_matches.argmax()
                        x1, y1, x2, y2 = bboxes_cls[pred_idx]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        true_positives += 1
                        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv.putText(image, f'{dataset["categories"][ic]["name"]}, {scores_cls[pred_idx]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

                    elif num_matches == 0:
                        false_negatives += 1

                        # Draw false negatives as blue
                        x1, y1, width, height = gt_bboxes[i]
                        x2 = x1 + width
                        y2 = y1 + height
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv.putText(image, f'{dataset["categories"][ic]["name"]}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)

                for i, pred_matches in enumerate(matches_matrix):
                    num_matches = pred_matches.sum()
                    if num_matches == 0:                    
                        # Draw false negatives as red
                        x1, y1, x2, y2 = bboxes_cls[i]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        false_positives += 1
                        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv.putText(image, f'{dataset["categories"][ic]["name"]}, {scores_cls[i]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)

            draw_text(image, true_positives, false_positives, false_negatives)
            all_true_positives += true_positives
            all_false_positives += false_positives
            all_false_negatives += false_negatives

            if false_positives == 0 and false_negatives == 0:
                outdir = 'only_TP'
            elif false_positives == 0 and false_negatives > 0:
                outdir = 'FN'
            elif false_positives > 0 and false_negatives == 0:
                outdir = 'FP'
            else:
                outdir = 'FP_FN'

            viz_path = f'{args.outpath}/test_images_with_prediction/{outdir}/{image_info["file_name"]}.jpg'
            cv.imwrite(viz_path, image)

        print(f"True positives: {all_true_positives}")
        print(f"False positives: {all_false_positives}")
        print(f"False negatives: {all_false_negatives}")




if __name__ == "__main__":
    '''
    Measure Average Precision and Average Recall for each class of a dataset, and find the score thresholds to optimize F1-score.
    Then use optimised score thresholds to determine and draw True Positives, False Positives and False Negatives on the test images    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, help="mmdetection checkpoint")
    parser.add_argument("--outpath", required=True, help="path to directory where to save results")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--pre_score_thr", type=float, default=0.2, help="pre-process score threshold")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()

    os.makedirs(f'{args.outpath}/test_images_with_prediction/', exist_ok=True)
    
    main(args)
