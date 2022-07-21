from mmdet.apis import init_detector, inference_detector
import numpy as np
import argparse
import json, os, glob
import imagesize
import torch
import mmcv
import tqdm
from utils import generate_saliency_map, parse_yolo_annotation, yolo_annotations_to_box, distance, bbox_to_yolo_annotations



def get_center(box):
    x1, y1, x2, y2 = box
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    
    return x_center, y_center



def make_overlap_box(box1, box2):
    x_center1, y_center1 = get_center(box1)
    x_center2, y_center2 = get_center(box2)
    
    if x_center1 < x_center2:
        x1 = x_center1
        x2 = x_center2
    else:
        x1 = x_center2
        x2 = x_center1

    if y_center1 < y_center2:
        y1 = y_center1
        y2 = y_center2
    else:
        y1 = y_center2
        y2 = y_center1

    return [x1, y1, x2, y2]



def make_overlap_boxes(target_boxes):
    '''
    Get boxes that cover diagonal defect:
    measure distance between top left and bottom right corners,
    and top right and bottom left corners.
    Then add an overlapping box for each pair of diagonal bboxes,
    whose coordinates are given by the centers of the two boxes
    '''
    new_target_boxes = []
    for boxes in target_boxes:
        if len(boxes) > 1:
            new_boxes = boxes.tolist()
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
                diag_ind = np.where(dist_matrix<50)[1]
                
                for ind in diag_ind:
                    if ind != i:
                        overlap_box = make_overlap_box(box, boxes[ind])
                        new_boxes.append(overlap_box)
                        
            new_boxes = np.unique(np.stack(new_boxes), axis=0)
            
        else:
            new_boxes = boxes
        new_target_boxes.append(new_boxes)
            
    return new_target_boxes



def main(args):
    '''
    Diagonal defects are annotated with bboxes "corner-to-corner". Add overlapping bboxes to each pair of such bboxes.
    '''
    # get list of images in directory
    images = glob.glob(f'{args.image_path}/*jpg')
    print(f'{len(images)} images in total')
    
    # get number of classes from classes.txt
    os.system(f'cp {args.annot_path}/classes.txt {args.output}/')
    with open(f'{args.output}/classes.txt', 'r') as f:
        n_class = len(f.readlines())
        
    # loop on image
    for im_path in tqdm.tqdm(images):
        annot_path = im_path.replace(args.image_path, args.annot_path).replace('.jpg', '.txt')
        annot_path_no_bracket = annot_path.replace('(', '\(').replace(')', '\)') # deal with annoying brackets in some filenames
        annot = parse_yolo_annotation(annot_path)

        if annot is None:
            os.system(f'cp {annot_path_no_bracket} {args.output}/')
            continue
        
        image_size = imagesize.get(im_path)
        target_boxes = yolo_annotations_to_box(annot, image_size, n_class)

        target_boxes_overlap = make_overlap_boxes(target_boxes) # add overlap boxes

        outname = im_path.replace(args.image_path, f'{args.output}/').replace('.jpg', '.txt')
        with open(outname, 'w') as f_out:
            for ic, boxes in enumerate(target_boxes_overlap):
                for box in boxes:
                    yolo_box = bbox_to_yolo_annotations(box, image_size)
                    f_out.write(f'{ic} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}')
                    f_out.write('\n')

        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="Yolo dataset directory containing .jpg")
    parser.add_argument("--annot-path", required=True, help="Yolo dataset directory containing .txt")

    parser.add_argument("--output", required=True, help="path to output annotations and images")

    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)

