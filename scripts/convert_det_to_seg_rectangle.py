import cv2
import numpy as np
import argparse
import json, os, glob
import tqdm
import imagesize
from utils import coco_to_box


def main(args):
    '''
    Make segmentation annotations using bboxes as segmentation masks.
    '''
    # get list of images in directory
    with open(args.annot_path, 'r') as f:
        annotations = json.load(f)
    print(f'{len(annotations["images"])} images in total')

    # get classes
    classes = []
    for cls in annotations['categories']:
        classes.append(cls['name'])

    data_path = '/'.join(args.annot_path.split('/')[:-1])
    
    # loop images
    for im in tqdm.tqdm(annotations['images']):
        im_path = f'{data_path}/{im["file_name"]}'
        size = imagesize.get(im_path)
        seg_map = np.zeros((size[1], size[0]))
        
        target_boxes = []
        for ann in annotations['annotations']:
            if ann['image_id'] == im['id']:
                box = coco_to_box(ann['bbox'])
                box.append((box[2]-box[0])*(box[3]-box[1])) # append surface
                box.append( ann['category_id'] ) # append class id
                target_boxes.append(box)

        # sort bboxes by surface (descending order)
        target_boxes = np.array(target_boxes)
        idx_sort = np.argsort(target_boxes[:,4])[::-1]
        sorted_boxes = target_boxes[idx_sort]
        
        # create segmentation mask from bboxes, starting from largest boxes
        
        # save cropped boxes and their coordinates
        for x1, y1, x2, y2, surface, class_id in sorted_boxes:
            seg_map[y1:y2, x1:x2] = class_id + 1
        seg_name = im_path.split('/')[-1].replace('.jpg', '.png')
        cv2.imwrite(f'{args.output}/{seg_name}', seg_map)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='path to output annotations and images')
    parser.add_argument('--annot-path', required=True, help='path to coco .json annotation file')

    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)

