from mmdet.apis import inference_detector
import numpy as np
import tqdm
import math
import cv2

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
        lines = f_in.readlines()
        
        if len(lines) == 0:
            return None
        
        for line in lines:
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
        x1 = int(round((annotation["x_center"]-annotation['width']/2) * image_width))
        if x1 < 0:
            x1 = 0
        y1 = int(round((annotation["y_center"]-annotation['height']/2) * image_height))
        if y1 < 0:
            y1 = 0
        x2 = int(round((annotation["x_center"]+annotation['width']/2) * image_width))
        y2 = int(round((annotation["y_center"]+annotation['height']/2) * image_height))
        box_annotations[annotation['class_index']].append([x1,y1,x2,y2])

    for c in range(n_class):
        if len(box_annotations[c]) > 0:
            box_annotations[c] = np.stack(box_annotations[c])

    return box_annotations
