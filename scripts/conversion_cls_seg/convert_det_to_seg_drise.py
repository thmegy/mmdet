from mmdet.apis import init_detector
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import argparse
import json
import os
import imagesize
import glob
import torch
import mmcv
from utils import generate_saliency_map, parse_yolo_annotation, yolo_annotations_to_box



def main(args):
    '''
    Convert annotations for detection task, in yolo format, to segmentation masks using D-RISE explanability method
    '''
    # get list of images in directory
    images = glob.glob(f'{args.image_path}/*jpg')
    image_size = imagesize.get(images[0])
    print('Image size = ', image_size)
    print(f'{len(images)} images in total')

    images_other_sizes = []
    for im in images:
        size = imagesize.get(im)
        if size[0] != image_size[0] or size[1] != image_size[1]:
            images_other_sizes.append(im)
    
    # get list of images already processed
    if not args.from_scratch:
        already_processed = glob.glob(f'{args.output}/*png')
        [images.remove(im.replace(args.output, args.image_path).replace('png', 'jpg')) for im in already_processed]
    print(f'{len(images)} images after removing already processed')

    # do not process images with no annotations
    no_annot = []
    for im in images:
        annot = parse_yolo_annotation(im.replace(args.image_path, args.annot_path).replace('.jpg', '.txt'))
        if annot is None:
            no_annot.append(im)

    for im in no_annot:
        cv2.imwrite( f'{args.output}/{im.split("/")[-1].replace(".jpg", ".png")}', np.zeros((image_size[1], image_size[0])) )
        images.remove(im)
        
    print(f'{len(images)} images after removing those without annotations')
    
    # get number of classes and image size for inference from mmdetection config
    config = mmcv.Config.fromfile(args.config)
    n_class = len(config.data.train.classes)

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
    # split images into batches to run the inference
    if len(images)==0:
        img_batches = []
    else:
        img_batch_size = args.batch_size
        img_n_batch = len(images) // img_batch_size
        if len(images) % img_batch_size != 0:
            img_n_batch += 1
        img_batches = np.array_split( np.array(images), img_n_batch )

    for batch in img_batches:
        images = []
        target_boxes = []
        for im in batch:
            image = cv2.imread(im)
            if image.shape[1] != image_size[0] and image.shape[0] != image_size[1]:
                image = cv2.resize(cv2.imread(im),
                                   None,
                                   fx=image_size[0] / image.shape[1],
                                   fy=image_size[1] / image.shape[0],
                                   interpolation=cv2.INTER_AREA)
            images.append(image)
            target_boxes.append( yolo_annotations_to_box(parse_yolo_annotation(im.replace(args.image_path, args.annot_path).replace('.jpg', '.txt')), image_size, n_class) )
        saliency_map = generate_saliency_map(model,
                                             images,
                                             image_size,
                                             n_class,
                                             target_boxes,
                                             prob_thresh=0.5,
                                             grid_size=(16, 16),
                                             n_masks=500)

        for im in range(len(images)):
            seg_map = np.zeros((image_size[1], image_size[0]))

            for ic in range(n_class):
                if len(target_boxes[im][ic]) == 0:
                    continue
                
                # keep segmentation map within annotated bboxes
                box_mask = np.full(saliency_map[im][ic][0].shape, False)
                for x1,y1,x2,y2 in target_boxes[im][ic]:
                    box_mask[y1:y2+1, x1:x2+1] = True

                for ib in range(len(target_boxes[im][ic])):
                    if saliency_map[im][ic][ib].sum() > 0:
                        score_mask = saliency_map[im][ic][ib]>np.percentile(saliency_map[im][ic][ib], 95)
                        segmentation_mask = score_mask & box_mask
                        
                        seg_map = np.where(segmentation_mask, ic+1, seg_map)
    
            cv2.imwrite(f'{args.output}/{batch[im].split("/")[-1].replace(".jpg", ".png")}', seg_map)

    # resize annotations for images with different sizes
    for im in images_other_sizes:
        size = imagesize.get(im)
        ann_file = f'{args.output}/{im.split("/")[-1].replace(".jpg", ".png")}'
        seg_annot = cv2.imread(ann_file)
        seg_annot = cv2.resize(seg_annot,
                   None,
                   fx= size[0] / seg_annot.shape[1],
                   fy= size[1] / seg_annot.shape[0],
                   interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(ann_file, cv2.cvtColor(seg_annot, cv2.COLOR_BGR2GRAY))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--model", required=True, help="mmdetection checkpoint")

    parser.add_argument("--image-path", required=True, help="Yolo dataset directory containing .jpg")
    parser.add_argument("--annot-path", required=True, help="Yolo dataset directory containing .txt")

    parser.add_argument("--output", required=True, help="path to output annotations and images")
    parser.add_argument("--from-scratch", action='store_true', help="Reprocess image already present in output directory")

    parser.add_argument('--batch-size', default=50, type=int, help='Number of images in inference batch')
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    main(args)

