import argparse
import mmdet.apis
import glob
import cv2 as cv
import os
import torch



#classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
classes = ('tronc',)


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    detector = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=device,
    )

    # get list of images in directory
    images = glob.glob(f'{args.im_dir}/*png')

    # perform inference
    predictions = mmdet.apis.inference_detector(detector, images)

    for im, preds in zip(images, predictions): # loop on images
        image = cv.imread(im)
        ann = []
        for ic, pred in enumerate(preds): # loop on classes
            for p in pred: # loop on bboxes
                x1, y1, x2, y2 = p[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if p[4] > args.score_threshold:
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv.putText(image, f'{classes[ic]}, {p[4]:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
                    ann.append(f'{x1} {y1} {x2} {y2}')
        outpath = im.replace(args.im_dir, args.viz_dir)
        cv.imwrite(outpath, image)
        with open(outpath.replace('.png', '.txt'), 'w') as f:
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
    args = parser.parse_args()

    os.makedirs(args.viz_dir, exist_ok=True)

    main(args)
