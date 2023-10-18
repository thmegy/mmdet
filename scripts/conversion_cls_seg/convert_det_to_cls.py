import cv2
import numpy as np
import argparse
import json, os, glob
import imagesize
import mmcv
import tqdm
from utils import parse_yolo_annotation, yolo_annotations_to_box, bbox_to_yolo_annotations, coco_to_box



def convert_yolo(args):
    '''
    Extract bboxes annotated in yolo format to create a classification database.
    '''
    # get list of images in directory
    images = glob.glob(f'{args.image_path}/*jpg')
    print(f'{len(images)} images in total')
    
    # get classes from mmdetection config
    with open(args.classes, 'r') as fcls:
        classes = [cls.replace('\n', '') for cls in fcls.readlines()]
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
                # require bboxes to have a minimal height and width --> avoids weird behaviours when resizing before training
                if (x2-x1) < 30:
                    x1 = x1 - 15
                    x2 = x2 + 15
                if (y2-y1) < 30:
                    y1 = y1 - 15
                    y2 = y2 + 15
                crop = im[y1:y2, x1:x2]
                crop_path = im_path.replace('.jpg', f'_{ib}.jpg').replace(args.image_path, f'{args.output}/{cls}/')
                try:
                    cv2.imwrite(crop_path, crop)
                except:
                    print(im_path)
                    print(x1, y1, x2, y2)

                # save cropped box coordinates
                with open(crop_path.replace('.jpg', '.txt'), 'w') as f_out:
                        f_out.write(f'{x1} {y1} {x2} {y2}')



def convert_coco(args):
    '''
    Extract bboxes annotated in coco format to create a classification database.
    '''
    # get list of images in directory
    with open(args.annot_path, 'r') as f:
        annotations = json.load(f)
    print(f'{len(annotations["images"])} images in total')

    # get classes
    classes = []
    for cls in annotations['categories']:
        classes.append(cls['name'])
        os.makedirs(f'{args.output}/{cls["name"]}/', exist_ok=True)

    data_path = '/'.join(args.annot_path.split('/')[:-1])
    
    # loop images
    for im in tqdm.tqdm(annotations['images']):
        target_boxes = [[] for _ in range(len(classes))]
        for ann in annotations['annotations']:
            if ann['image_id'] == im['id']:
                box = coco_to_box(ann['bbox'])
                cat = ann['category_id']
                target_boxes[cat].append(box)
        
        im_path = f'{data_path}/{im["file_name"]}'
        im = cv2.imread(im_path)
        
        # save cropped boxes and their coordinates
        for cls, boxes in zip(classes, target_boxes):
            for ib, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # require bboxes to have a minimal height and width --> avoids weird behaviours when resizing before training
                if (x2-x1) < 30:
                    x1 = x1 - 15
                    x2 = x2 + 15
                if (y2-y1) < 30:
                    y1 = y1 - 15
                    y2 = y2 + 15

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0

                crop = im[y1:y2, x1:x2]
                crop_name =  im_path.split('/')[-1].replace('.jpg', f'_{ib}.jpg')
                crop_path = f'{args.output}/{cls}/{crop_name}'
                try:
                    cv2.imwrite(crop_path, crop)
                except:
                    print(im_path)
                    print(x1, y1, x2, y2)

                # save cropped box coordinates
                with open(crop_path.replace('.jpg', '.txt'), 'w') as f_out:
                        f_out.write(f'{x1} {y1} {x2} {y2}')
                        
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='path to output annotations and images')
    subparsers = parser.add_subparsers(dest='format', help='Annotation format')

    # yolo arguments
    parser_yolo = subparsers.add_parser('yolo')
    parser_yolo.add_argument('--classes', required=True, help='path to classes file')
    parser_yolo.add_argument('--image-path', required=True, help='Yolo dataset directory containing .jpg')
    parser_yolo.add_argument('--annot-path', required=True, help='Yolo dataset directory containing .txt')

    # coco arguments
    parser_coco = subparsers.add_parser('coco')
    parser_coco.add_argument('--annot-path', required=True, help='path to coco .json annotation file')


    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)

    if args.format == 'yolo':
        convert_yolo(args)
    elif args.format == 'coco':
        convert_coco(args)

