import cv2
import numpy as np
import argparse
import json, os, glob
import imagesize
import mmcv
import tqdm
from utils import parse_yolo_annotation, yolo_annotations_to_box, bbox_to_yolo_annotations



def main(args):
    '''
    Extract bboxes annotated in yolo format to create a classification database.
    '''
    # get list of images in directory
    images = glob.glob(f'{args.image_path}/*jpg')
    print(f'{len(images)} images in total')
    
    # get classes from mmdetection config
    config = mmcv.Config.fromfile(args.config)
    classes = config.data.train.classes
    for cls in classes:
        os.makedirs(f'{args.output}/{cls}/', exist_ok=True)

    
    # loop images
    for im_path in tqdm.tqdm(images):
        annot_path = im_path.replace(args.image_path, args.annot_path).replace('.jpg', '.txt')
        annot_path_no_bracket = annot_path.replace('(', '\(').replace(')', '\)') # deal with annoying brackets in some filenames
        annot = parse_yolo_annotation(annot_path)

        if annot is None:
            continue

        image_size = imagesize.get(im_path)
        target_boxes = yolo_annotations_to_box(annot, image_size, len(classes))

        im = cv2.imread(im_path)
        
        # save cropped boxes and their coordinates
        for cls, boxes in zip(classes, target_boxes):
            for ib, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                crop = im[y1:y2, x1:x2]
                crop_path = im_path.replace('.jpg', f'_{ib}.jpg').replace(args.image_path, f'{args.output}/{cls}/')
                cv2.imwrite(crop_path, crop)

                # save cropped box coordinates
                with open(crop_path.replace('.jpg', '.txt'), 'w') as f_out:
                        f_out.write(f'{x1} {y1} {x2} {y2}')

        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")

    parser.add_argument("--image-path", required=True, help="Yolo dataset directory containing .jpg")
    parser.add_argument("--annot-path", required=True, help="Yolo dataset directory containing .txt")

    parser.add_argument("--output", required=True, help="path to output annotations and images")

    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)

