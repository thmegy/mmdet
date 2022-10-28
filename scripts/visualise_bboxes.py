import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cv2
import tqdm
import imagesize
from utils import parse_yolo_annotation, yolo_annotations_to_box



colors_list = [[1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,0], [1,0,1], [0.5,1,0], [0,0.5,1], [1,0.5,1], [0.5,1,1], [1,1,0.5]]


def main(args):
    '''
    Visualise annotated bounding boxes on original images
    '''
    
    # get names of images from detection dataset
    images = glob.glob(f'{args.image_path}/*.jpg')

    # get number of classes from classes.txt
    os.system(f'cp {args.annot_path}/classes.txt {args.output}/')
    with open(f'{args.output}/classes.txt', 'r') as f:
        n_class = len(f.readlines())

    print(n_class)
        
    for im_path in tqdm.tqdm(images):
        annot_path = im_path.replace(args.image_path, args.annot_path).replace('.jpg', '.txt')
        annot_path_no_bracket = annot_path.replace('(', '\(').replace(')', '\)') # deal with annoying brackets in some filenames
        annot = parse_yolo_annotation(annot_path)
        
        if annot is None:
            continue

        image_size = imagesize.get(im_path)
        target_boxes = yolo_annotations_to_box(annot, image_size, n_class)

#        if len(target_boxes[7]) == 0:
#            continue
        
        image = cv2.imread(im_path)
        
        for ic in range(n_class):
            for i in range(len(target_boxes[ic])):
                cv2.rectangle(image, target_boxes[ic][i,:2], target_boxes[ic][i, 2:],
                              (np.flip(np.array(colors_list[ic])*255)).tolist(), 5)

        cv2.imwrite(im_path.replace(args.image_path, args.output), image)
           


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="Yolo dataset directory containing .jpg")
    parser.add_argument("--annot-path", required=True, help="Yolo dataset directory containing .txt")

    parser.add_argument("--output", required=True, help="path to output images")
    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)

