from mmdet.apis import init_detector, inference_detector
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import argparse
import json
import os
import imagesize
import torch
import mmcv
from utils import generate_saliency_map



def main(args):
    '''
    Use D-RISE explanability method to find what pixels were important to predict bboxes in an image.
    '''
    # get number of classes and image size for inference from mmdetection config
    config = mmcv.Config.fromfile(args.config)
    classes = config.data.train.classes
    n_class = len(classes)

    # load model
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = init_detector(args.config, args.model, device)

    # run inference on image: get predicted bbox
    image = cv2.imread(args.image)
    out = inference_detector(model, image)
    pred_boxes = [[] for _ in range(n_class)]
    pred_scores = [[] for _ in range(n_class)]
    for i, pred in enumerate(out):
        for *box, score in pred:
            if score < args.score_threshold:
                continue
            box = tuple(np.round(box).astype(int).tolist())
            print(i, classes[i], box, score)
            pred_boxes[i].append(box)
            pred_scores[i].append(score)
        if len(pred_boxes[i]) > 0:
            pred_boxes[i] = np.stack(pred_boxes[i])

    # get saliency map
    saliency_map = generate_saliency_map(model,
                                         [image], # need to be a list of images, even if only one
                                         (image.shape[1], image.shape[0]),
                                         n_class,
                                         [pred_boxes], # need to be a list of images, even if only one
                                         prob_thresh=0.5,
                                         grid_size=(25, 25),
                                         n_masks=600)

    saliency_map = saliency_map[0] # only one image
    
    for ic, boxes in enumerate(pred_boxes):
        for ib in range(len(boxes)):
            image_with_bbox = image.copy()
            cv2.rectangle(image_with_bbox, tuple(boxes[ib, :2]),
                          tuple(boxes[ib, 2:]), (255, 0, 0), 3)

            plt.figure(figsize=(12, 8))
            plt.imshow(image_with_bbox)

            plt.imshow(saliency_map[ic][ib], cmap='jet', alpha=0.3)
            plt.text(0, 0, f'{classes[ic]} {pred_scores[ic][ib]:.3f}', fontsize=20, color='red', va='bottom', ha='left')
            
            plt.axis('off')
            plt.savefig(f'{args.output}/{args.image.split("/")[-1].replace(".jpg", "")}_{classes[ic]}_{ib}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--model", required=True, help="mmdetection checkpoint")
    parser.add_argument("--image", required=True, help="path to image")
    parser.add_argument("--output", required=True, help="path to output annotations and images")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="bbox score threshold")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    main(args)

