import argparse
import mmdet.apis
import glob
import cv2 as cv
import os
import json
import numpy as np
import tqdm



def get_pred_numbers(predictions, threshold):
    classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    num_dict = {}
    for ic, pred in enumerate(predictions): # loop on classes
        mask = pred[:,4] > threshold # filter bboxes
        pred = pred[mask][:,:4].astype(int)
        pred_list = []
        for p in pred: # loop on bboxes
            x1, y1, x2, y2 = p
            pred_list.append(p.tolist())
        if len(pred_list) > 0:
            num_dict[str(classes[ic])] = pred_list
    return num_dict



def main(args):
    # get list of images to run inference on
    with open(args.dataset, "rt") as f_in:
        dataset = json.load(f_in)
    images = [ f'{args.im_dir}/{im["file_name"]}' for im in dataset['images'] ]
#    images = glob.glob(f'{args.im_dir}/*jpg')

    # build plaque detector
    detector_plaque = mmdet.apis.init_detector(
        args.config[0],
        args.checkpoint[0],
        device=f"cuda:{args.gpu_id}",
    )

    # perform inference --> find plaques
    # split images into batches to run the inference
    img_batch_size = 30
    img_n_batch = len(images) // img_batch_size
    img_batches = np.array_split( np.array(images), img_n_batch )
    predictions = []
    print('\nRunning 1st inference --> find plaques')
    for img_batch in tqdm.tqdm(img_batches):
        predictions = predictions + mmdet.apis.inference_detector(detector_plaque, img_batch.tolist())


    # remove first detector and build number detector
    del detector_plaque
    detector_number = mmdet.apis.init_detector(
        args.config[1],
        args.checkpoint[1],
        device=f"cuda:{args.gpu_id}",
    )

    # perform inference with number detector on plaques found in the first inference
    out_dict = {}
    print('\nRunning 2nd inference --> find numbers')
    for im, preds in tqdm.tqdm( zip(images, predictions) ): # loop on images
        image = cv.imread(im)
        pred = preds[0]
        mask = pred[:,4] > args.score_threshold # filter bboxes
        pred = pred[mask][:,:4].astype(int)
        pred_list = []
        if pred.shape[0] > 0:
            for ip, p in enumerate(pred): # loop on bboxes
                x1, y1, x2, y2 = p
                cropped_image = image[y1:y2, x1:x2] # crop image to keep only bbox around plaque
                cropped_image_path = im.replace(args.im_dir, f'{args.viz_dir}/cropped/').replace('.jpg', f'_crop_{ip}.jpg')
                cv.imwrite(cropped_image_path, cropped_image)
            
                pred_numbers = mmdet.apis.inference_detector(detector_number, cropped_image_path) # perform inference --> find numbers
                pred_numbers_filter = get_pred_numbers(pred_numbers, args.score_threshold)
                pred_list.append( {'bbox':p.tolist(), 'numbers':pred_numbers_filter} )

                cv.rectangle(image, (x1, y1), (x2, y2), (20, 46, 209), 2)
                # draw number bboxes
                for num, bbox_list in pred_numbers_filter.items():
                    for coord in bbox_list:
                        cv.rectangle(image, (x1+coord[0], y1+coord[1]), (x1+coord[2], y1+coord[3]), (38, 105, 38), 2)
                        cv.putText(image, num, (x1+coord[0], y1+coord[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, (38, 105, 38), 2, cv.LINE_AA)

            cv.imwrite(im.replace(args.im_dir, args.viz_dir), image)
            cropped_image = image[int(0.9*y1):int(1.1*y2), int(0.9*x1):int(1.1*x2)] # take some margin around bbox
            cropped_image_path = im.replace(args.im_dir, f'{args.viz_dir}/cropped/').replace('.jpg', f'_crop_{ip}.jpg')
            cv.imwrite(cropped_image_path, cropped_image)
            out_dict[im.replace(args.im_dir, '')] = pred_list
        
    with open(f'{args.viz_dir}/results.json', 'w') as out_f:
        json.dump(out_dict, out_f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs='*', help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, nargs='*', help="mmdetection checkpoint")
    parser.add_argument("--dataset", required=True, help="JSON cocolike dataset")
    parser.add_argument("--im-dir", required=True, help="Directory containing the images")
    parser.add_argument("--viz-dir", required=True, help="Directory where visualizations will be saved")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="bbox score threshold")
    args = parser.parse_args()

    os.makedirs(f'{args.viz_dir}/cropped/', exist_ok=True)

    main(args)
