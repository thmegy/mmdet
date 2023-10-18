import argparse
import mmdet.apis
import glob
import cv2 as cv
import os
import torch
import mmengine
import tqdm
import numpy as np
import json

threshold_dict = {
    'Arrachement_pelade':0.27,
    'Faiencage':0.35,
    'Nid_de_poule':0.41,
    'Transversale':0.36,
    'Longitudinale':0.41,
    'Pontage_de_fissures':0.47,
    'Remblaiement_de_tranchees':0.44,
    'Raccord_de_chaussee':0.35,
    'Comblage_de_trou_ou_Projection_d_enrobe':0.30,
    'Bouche_a_clef':0.35,
    'Grille_avaloir':0.35,
    'Regard_tampon':0.35,

    }

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    config = mmengine.Config.fromfile(args.config)
    classes = config.classes

    detector = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=device,
    )

    # get list of images in directory
    images = glob.glob(f'{args.im_dir}/*png')

    for im in tqdm.tqdm(images):
        preds = mmdet.apis.inference_detector(detector, im) # run inference
        image = cv.imread(im)
        outpath = im.replace(args.im_dir, args.viz_dir)
        ann = []

        bboxes = preds.pred_instances.numpy()['bboxes']
        scores = preds.pred_instances.numpy()['scores']
        labels = preds.pred_instances.numpy()['labels']

        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#            if score > args.score_threshold:
            if score > threshold_dict[classes[label]]:
                if args.save_image:
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
                    cv.putText(image, f'{classes[label]}, {score:.2f}', (x1 ,y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
                if args.yolo_format:
                    image_width = image.shape[1]
                    image_height = image.shape[0]
                    ann.append(f'{label} {(x1+x2)/2/image_width:.3f} {(y1+y2)/2/image_height:.3f} {(x2-x1)/image_width:.3f} {(y2-y1)/image_height:.3f}')
                else:
                    ann.append(f'{x1} {y1} {x2} {y2}')                            

        if args.save_image:
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
    parser.add_argument("--score-threshold", type=float, default=0.15, help="bbox score threshold")
    parser.add_argument("--yolo-format", action='store_true', help="yolo format for saved annotations")
    parser.add_argument("--save-image", action='store_true', help="saved image with draw bboxes")
    args = parser.parse_args()

    os.makedirs(args.viz_dir, exist_ok=True)

    main(args)
