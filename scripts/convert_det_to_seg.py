from mmdet.apis import init_detector, inference_detector
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import matplotlib.colors as mcolors
import argparse
import json
import os
import imagesize
import glob
import torch
import mmcv



def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask



def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked



def iou(boxes, box):
    box = np.asarray(box)
    tl = np.maximum(boxes[:,:2], box[:2])
    br = np.minimum(boxes[:,2:], box[2:])
    intersection = np.prod(br - tl, axis=1) * np.all(tl < br, axis=1).astype(float)
    area1 = np.prod(boxes[:,2:] - boxes[:,:2], axis=1)
    area2 = np.prod(box[2:] - box[:2])
    return intersection / (area1 + area2 - intersection)



def generate_saliency_map(model,
                          images,
                          image_size,
                          n_classes,
                          target_boxes,
                          prob_thresh=0.5,
                          grid_size=(16, 16),
                          n_masks=5000,
                          seed=0):
    np.random.seed(seed)
    image_w, image_h = image_size
    res = [[[np.zeros((image_h, image_w), dtype=np.float32) for _ in range(len(target_boxes[im][ic]))] for ic in range(n_classes)] for im in range(len(images))]

    for i in tqdm.tqdm(range(n_masks)):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        masked = [mask_image(im, mask) for im in images]
        out = inference_detector(model, masked)
        for im in range(len(images)):
            for ic in range(n_classes):
                boxes = target_boxes[im][ic]
                pred = out[im][ic]
                if len(pred) > 0 and len(boxes) > 0:
                    score = np.stack([iou(boxes, box) * score for *box, score in pred]).max(axis=0)
                    for ib in range(len(target_boxes[im][ic])):
                        res[im][ic][ib] += mask * score[ib]
    return res



def parse_yolo_annotation(annotation_txtpath):
    """ Parse a yolo annotation file. """
    annotations = []
    with open(annotation_txtpath, "rt") as f_in:
        for line in f_in.readlines():
            line = line.strip()

            cls, x_center, y_center, width, height = line.split()

            annotations.append(
                {
                    "class_index": int(cls),
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width": float(width),
                    "height": float(height),
                }
            )
    return annotations



def yolo_annotations_to_box(yolo_annotations, image_size, n_class):
    """ Convert a yolo annotation list to (x1, y1, x2, y2) coordinates."""
    image_width = image_size[0]
    image_height = image_size[1]
    box_annotations = [[] for _ in range(n_class)]

    for annotation in yolo_annotations:
        x1 = int(round((annotation["x_center"]-annotation['width']) * image_width))
        if x1 < 0:
            x1 = 0
        y1 = int(round((annotation["y_center"]-annotation['height']) * image_height))
        if y1 < 0:
            y1 = 0
        x2 = int(round((annotation["x_center"]+annotation['width']) * image_width))
        y2 = int(round((annotation["y_center"]+annotation['height']) * image_height))
        box_annotations[annotation['class_index']].append([x1,y1,x2,y2])

    for c in range(n_class):
        if len(box_annotations[c]) > 0:
            box_annotations[c] = np.stack(box_annotations[c])

    return box_annotations



def main(args):
    '''
    Convert annotations for detection task, in yolo format, to segmentation masks using D-RISE explanability method
    '''
    # get list of images in directory
    images = glob.glob(f'{args.image_path}/*jpg')
    image_size = imagesize.get(images[0])

    # get number of classes and image size for inference from mmdetection config
    config = mmcv.Config.fromfile(args.config)
    n_class = len(config.data.train.classes)
    new_image_size = config.data.test.pipeline[1]['img_scale']

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
    img_batch_size = args.batch_size
    img_n_batch = len(images) // img_batch_size
    img_batches = np.array_split( np.array(images), img_n_batch )

    for batch in img_batches:
        images = [cv2.resize(cv2.imread(im),
                             None,
                             fx= new_image_size[0] / image_size[0],
                             fy= new_image_size[1] / image_size[1],
                             interpolation=cv2.INTER_AREA)
                  for im in batch]
        target_boxes = [yolo_annotations_to_box(parse_yolo_annotation(im.replace(args.image_path, args.annot_path).replace('.jpg', '.txt')), new_image_size, n_class) for im in batch]
        saliency_map = generate_saliency_map(model,
                                             images,
                                             new_image_size,
                                             n_class,
                                             target_boxes,
                                             prob_thresh=0.5,
                                             grid_size=(16, 16),
                                             n_masks=800)

        for im in range(len(batch)):
            image_with_bbox = images[im].copy()
            for ic in range(n_class):
                for i in range(len(target_boxes[im][ic])):
                    cv2.rectangle(image_with_bbox, tuple(target_boxes[im][ic][i,:2]),
                                  tuple(target_boxes[im][ic][i,2:]), (np.flip(np.array(colors_list[ic])*255)).tolist(), 5)
        
            plt.figure(figsize=(12, 12))
            plt.imshow(image_with_bbox[:, :, ::-1])

            for ic in range(n_class):
                for ib in range(len(target_boxes[im][ic])):
                    if saliency_map[im][ic][ib].sum() > 0:
                        box_mask = np.full(saliency_map[im][ic][ib].shape, False)
                        x1,y1,x2,y2 = target_boxes[im][ic][ib]
                        box_mask[y1:y2+1, x1:x2+1] = True
                        score_mask = saliency_map[im][ic][ib]>np.percentile(saliency_map[im][ic][ib], 95)
                        segmentation_mask = score_mask & box_mask

                        plt.imshow(segmentation_mask, cmap=cmaps[ic], alpha=0.3)
    
            plt.axis('off')
            plt.savefig(f'{args.output}/{batch[im].split("/")[-1]}')
            
#    with open(args.output, "wt") as f_out:
#        json.dump(output_coco, f_out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--model", required=True, help="mmdetection checkpoint")

    parser.add_argument("--image-path", required=True, help="Yolo dataset directory containing .jpg")
    parser.add_argument("--annot-path", required=True, help="Yolo dataset directory containing .txt")

    parser.add_argument("--output", required=True, help="path to output annotations and images")

    parser.add_argument('--batch-size', default=50, type=int, help='Number of images in inference batch')
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    main(args)

