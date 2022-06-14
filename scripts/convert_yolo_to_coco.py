import argparse
import json
import os

import imagesize


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


def yolo_annotations_to_coco(yolo_annotations, image_size):
    """ Convert a yolo annotation list to a coco annotation list."""
    image_width, image_height = image_size
    coco_annotations = []
    for annotation in yolo_annotations:
        x_center = int(round(annotation["x_center"] * image_width))
        y_center = int(round(annotation["y_center"] * image_height))
        width = int(round(annotation["width"] * image_width))
        height = int(round(annotation["height"] * image_height))

        # XXX: Odd widths and heights may end up half a pixel lower because of //
        min_x = x_center - width // 2
        min_y = y_center - height // 2

        coco_annotations.append(
            {
                "class_index": annotation["class_index"],
                "min_x": min_x,
                "min_y": min_y,
                "width": width,
                "height": height,
            }
        )
    return coco_annotations


def parse_yolo_classes(classe_filepath):
    """ Read a yolo class files and return class names."""
    with open(args.classes, "rt") as f_in:
        class_names = f_in.readlines()
    class_names = [name.strip() for name in class_names]
    return class_names


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        required=True,
        help="Yolo dataset directory containing .jpg",
    )
    parser.add_argument(
        "--annot-path",
        required=True,
        help="Yolo dataset directory containing .txt",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help="File containing the class names ('obj.names' or 'classes.txt')",
    )
    parser.add_argument("--output", required=True, help="Output .json file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    image_filenames = []
    for entry in os.scandir(args.image_path):
        name, ext = os.path.splitext(entry.name)
        if ext not in [".jpg", ".png"]:
            continue
        image_filenames.append(entry.name)
    image_filenames = sorted(image_filenames)

    annot_filenames = [os.path.splitext(fname)[0] + ".txt" for fname in image_filenames]

    output_coco = {"images": [], "annotations": [], "categories": []}

    annotation_count = 0
    for idx, (image_fname, annot_fname) in enumerate(
        zip(image_filenames, annot_filenames)
    ):
        annot_path = os.path.join(args.annot_path, annot_fname)
        yolo_annotations = parse_yolo_annotation(annot_path)

        image_path = os.path.join(args.image_path, image_fname)
        image_size = imagesize.get(image_path)

        coco_annotations = yolo_annotations_to_coco(yolo_annotations, image_size)

        output_coco["images"].append(
            {
                "id": idx,
                "file_name": image_fname,
                "width": image_size[0],
                "height": image_size[1],
            }
        )

        for coco_annotation in coco_annotations:
            output_coco["annotations"].append(
                {
                    "id": annotation_count,
                    "image_id": idx,
                    "category_id": coco_annotation["class_index"],
                    "bbox": [
                        coco_annotation["min_x"],
                        coco_annotation["min_y"],
                        coco_annotation["width"],
                        coco_annotation["height"],
                    ],
                    "area": coco_annotation["width"] * coco_annotation["height"],
                    "iscrowd": False,
                }
            )
            annotation_count += 1

    class_names = parse_yolo_classes(args.classes)
    for i, class_name in enumerate(class_names):
        output_coco["categories"].append(
            {"id": i, "name": class_name,}
        )

    with open(args.output, "wt") as f_out:
        json.dump(output_coco, f_out)
