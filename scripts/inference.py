import argparse
import mmdet.apis
import glob
import cv2 as cv
import os
import torch
import mmcv
import tqdm
import numpy as np
import json


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    config = mmcv.Config.fromfile(args.config)
    classes = config.classes
    
    detector = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=device,
    )

    # get list of images in directory
    images = glob.glob(f'{args.im_dir}/*jpg')

    # perform inference
    # split images into batches to run the inference
    img_batch_size = 3
    img_n_batch = len(images) // img_batch_size
    img_batches = np.array_split( np.array(images), img_n_batch )
    predictions = []
    for img_batch in tqdm.tqdm(img_batches):
        predictions = predictions + mmdet.apis.inference_detector(detector, img_batch.tolist())

    
    for im, preds in tqdm.tqdm(zip(images, predictions)): # loop on images
        image = cv.imread(im)
        outpath = im.replace(args.im_dir, args.viz_dir)
        ann = []
        
        if args.is_seg:
            color = np.array([0,255,0], dtype='uint8') # mask color
            mask_full = np.full(image.shape[:2], False) # overall mask to gather all instance masks
            
            for ic, (pred, seg) in enumerate(zip(preds[0], preds[1])): # loop on classes
                for ip, (p, mask) in enumerate(zip(pred, seg)): # loop on instances (bbox + segmentation mask)
                    x1, y1, x2, y2 = p[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if p[4] > args.score_threshold:
                        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv.putText(image, f'{classes[ic]}, {p[4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
                        ann.append(f'{x1} {y1} {x2} {y2}')

                        mask_full += mask

            masked_image = np.where(mask_full[...,None], color, image)
            cv.imwrite(outpath, masked_image)

            # save segmentation masks
            with open(outpath.replace('.png', '.npy'), 'wb') as f:
                np.save(f, mask_full)
                    
        else:
            for ic, pred in enumerate(preds): # loop on classes
                ann_idx = []
                for ip, p in enumerate(pred): # loop on bboxes
                    x1, y1, x2, y2 = p[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if p[4] > args.score_threshold:
                        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv.putText(image, f'{classes[ic]}, {p[4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
                        if args.yolo_format:
                            image_width = image.shape[1]
                            image_height = image.shape[0]
                            ann.append(f'{ic} {(x1+x2)/2/image_width} {(y1+y2)/2/image_height} {(x2-x1)/image_width} {(y2-y1)/image_height}')
                        else:
                            ann.append(f'{x1} {y1} {x2} {y2}')                            
            
            cv.imwrite(outpath, image)

            
        with open(outpath.replace('.jpg', '.txt'), 'w') as f:
            for a in ann:
                f.write(f'{a}\n')

        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, help="mmdetection checkpoint")
    parser.add_argument("--im-dir", required=True, help="Directory containing the images")
    parser.add_argument("--viz-dir", required=True, help="Directory where visualizations will be saved")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="bbox score threshold")
    parser.add_argument("--is-seg", action='store_true', help="Instance segmentation network, mask is expected in predictions")
    parser.add_argument("--yolo-format", action='store_true', help="yolo format for saved annotations")
    args = parser.parse_args()

    os.makedirs(args.viz_dir, exist_ok=True)

    main(args)
