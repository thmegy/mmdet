from mmdet.apis import init_detector, inference_detector
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import argparse
import json, os, glob
import imagesize
import torch
import mmcv
import tqdm
from utils import generate_saliency_map, parse_yolo_annotation, yolo_annotations_to_box, distance, bbox_to_yolo_annotations



def crop_inference(model, image, ic, score_threshold, shift):
    pred_boxes = []
    pred_scores = []
    pred = inference_detector(model, image)[ic]
    for *box, score in pred:
        if score < score_threshold:
            continue
        box = np.round(box).astype(int).tolist()
        box[0] += shift[0]
        box[1] += shift[1]
        box[2] += shift[0]
        box[3] += shift[1]
        pred_boxes.append(box)
        pred_scores.append(score)
    return pred_boxes, pred_scores



def nms(boxes, scores, thresh):
    '''
    boxes is a numpy array : num_boxes, 4
    scores ia  nump array : num_boxes,
    '''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



def get_boxes_diagonal(target_boxes):
    '''
    Get boxes that cover diagonal defect:
    measure distance between top left and bottom right corners,
    and top right and bottom left corners
    '''
    boxes_diagonal = []
    other_boxes = []
    for boxes in target_boxes:
        if len(boxes) > 1:
            diag_ind = []
            other_ind = []
            for i, box in enumerate(boxes):
                dist_matrix = []
                
                box_bottom_right = box[[2,3]]
                boxes_top_left = boxes[:,[0,1]]
                dist_matrix.append(distance(boxes_top_left, box_bottom_right))
                
                box_top_right = box[[2,1]]
                boxes_bottom_left = boxes[:,[0,3]]
                dist_matrix.append(distance(boxes_bottom_left, box_top_right))

                boxes_bottom_right = boxes[:,[2,3]]
                box_top_left = box[[0,1]]
                dist_matrix.append(distance(boxes_bottom_right, box_top_left))
                
                boxes_top_right = boxes[:,[2,1]]
                box_bottom_left = box[[0,3]]
                dist_matrix.append(distance(boxes_top_right, box_bottom_left))
                
                dist_matrix = np.array(dist_matrix)
                if np.where(dist_matrix<50, True, False).sum() > 0:
                    diag_ind.append(i)
                else:
                    other_ind.append(i)
                    
            if len(diag_ind) > 0:
                boxes_diagonal.append( np.stack(boxes[diag_ind]) )
            else:
                boxes_diagonal.append([])
                
            if len(other_ind) > 0:
                other_boxes.append( np.stack(boxes[other_ind]) )
            else:
                other_boxes.append([])
                
        elif len(boxes)==1:
            boxes_diagonal.append([])
            other_boxes.append(boxes)
        else:
            boxes_diagonal.append([])
            other_boxes.append([])
            
    return boxes_diagonal, other_boxes



def main(args):
    '''
    Convert large overlapping annotated bboxes (eg for diagonal cracks) into many smaller overlapping bboxes.
    '''
    # get list of images in directory
    images = glob.glob(f'{args.image_path}/*jpg')
    print(f'{len(images)} images in total')
    
    # get number of classes from mmdetection config
    config = mmcv.Config.fromfile(args.config)
    n_class = len(config.data.train.classes)
    with open(f'{args.output}/modified/classes.txt', 'w') as f:
        for c in config.data.train.classes:
            f.write(c)
            f.write('\n')

    # creat color maps
    colors_list = [[1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,0]]
    cmaps = []
    for ic in range(n_class):
        colors = [(colors_list[ic][0], colors_list[ic][1], colors_list[ic][2], c) for c in np.linspace(0,1,100)]
        cmaps.append( mcolors.LinearSegmentedColormap.from_list(f'mycmap{ic}', colors, N=5) )

    # load model
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = init_detector(args.config, args.model, device)

    # perform inference
    for im_path in tqdm.tqdm(images):
        annot_path = im_path.replace(args.image_path, args.annot_path).replace('.jpg', '.txt')
        annot_path_no_bracket = annot_path.replace('(', '\(').replace(')', '\)') # deal with annoying brackets in some filenames
        annot = parse_yolo_annotation(annot_path)

        if annot is None:
            os.system(f'cp {annot_path_no_bracket} {args.output}/not_modified/')
            continue
        
        image_size = imagesize.get(im_path)
        target_boxes = yolo_annotations_to_box(annot, image_size, n_class)
        boxes_diagonal, other_boxes = get_boxes_diagonal(target_boxes)
        if sum([len(b) for b in boxes_diagonal]) == 0:
            os.system(f'cp {annot_path_no_bracket} {args.output}/not_modified/')
            continue            

        im_path_no_bracket = im_path.replace('(', '\(').replace(')', '\)')  # deal with annoying brackets in some filenames
        os.system(f'cp {im_path_no_bracket} {args.output}/modified/')
        pred_boxes = [[] for _ in range(n_class)]
        pred_scores = [[] for _ in range(n_class)]
        boxes_diagonal_leftover = [[] for _ in range(n_class)]
        new_annot = []

        im = cv2.imread(im_path)
        
        # run inference on cropped bboxes
        for ic, boxes in enumerate(boxes_diagonal):
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = im[y1:y2, x1:x2]
                pred = crop_inference(model, crop, ic, args.score_threshold, (x1, y1))
                pred_boxes[ic] += pred[0]
                pred_scores[ic] += pred[1]
                if len(pred[0]) == 0:
                    boxes_diagonal_leftover[ic].append(box)
        
            if len(pred_boxes[ic]) > 0:       
                pred_boxes[ic] = np.stack(pred_boxes[ic])

        # nms on newly predicted bboxes then save to new annotation file
        for i, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
            # boxes with overlap
            if len(boxes) > 0:
                boxes = np.stack(boxes)
                scores = np.stack(scores)
                keep_ind = nms(boxes, scores, args.iou_threshold)
                pred_boxes[i] = pred_boxes[i][keep_ind]
        
                for box in pred_boxes[i]:
                    yolo_box = bbox_to_yolo_annotations(box, image_size)
                    new_annot.append(f'{i} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}')
                    
            # boxes with overlap but no prediction
            for box in boxes_diagonal_leftover[i]:
                yolo_box = bbox_to_yolo_annotations(box, image_size)
                new_annot.append(f'{i} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}')
                    
            # boxes without overlap
            for box in other_boxes[i]:
                yolo_box = bbox_to_yolo_annotations(box, image_size)
                new_annot.append(f'{i} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}')

        outname = im_path.replace(args.image_path, f'{args.output}/modified/').replace('.jpg', '.txt')
        with open(outname, 'w') as f_out:
            for ann in new_annot:
                f_out.write(ann)
                f_out.write('\n')

        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--model", required=True, help="mmdetection checkpoint")

    parser.add_argument("--image-path", required=True, help="Yolo dataset directory containing .jpg")
    parser.add_argument("--annot-path", required=True, help="Yolo dataset directory containing .txt")

    parser.add_argument("--output", required=True, help="path to output annotations and images")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")

    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.35, help="iou threshold")
    
    args = parser.parse_args()

    os.makedirs(f'{args.output}/modified/', exist_ok=True)
    os.makedirs(f'{args.output}/not_modified/', exist_ok=True)
    
    main(args)

