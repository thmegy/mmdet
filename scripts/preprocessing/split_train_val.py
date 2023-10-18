import argparse
import json
import random


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original", required=True, help="Original dataset (not split)"
    )
    parser.add_argument("--train", required=True, help="Train dataset (after split)")
    parser.add_argument("--val", required=True, help="Validation dataset (after split)")
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="The validation will be ratio * original examples.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.original, "rt") as f_in:
        annotations = json.load(f_in)

    print(f"There are {len(annotations['images'])} images.")

    count_per_category = {cat["name"]: 0 for cat in annotations["categories"]}
    for annot in annotations["annotations"]:
        cur_category_name = annotations["categories"][annot["category_id"]]["name"]
        count_per_category[cur_category_name] += 1

    print("")
    print("Counts per category:")
    for category, count in count_per_category.items():
        print(f"{category} : {count}")

    was_success = False
    for i in range(1000):
        # Selected random images from the dataset to be part of validation
        val_indices = set()
        for image in annotations["images"]:
            if random.random() < args.ratio:
                val_indices.add(image["id"])

        annotations_val = {
            "images": [],
            "annotations": [],
            "categories": annotations["categories"],
        }
        annotations_train = {
            "images": [],
            "annotations": [],
            "categories": annotations["categories"],
        }

        count_per_category_val = {cat["name"]: 0 for cat in annotations["categories"]}

        # Pick annotations corresponding to selected images
        for annot in annotations["annotations"]:
            if annot["image_id"] in val_indices:
                annotations_val["annotations"].append(annot)
                cur_category_name = annotations["categories"][annot["category_id"]][
                    "name"
                ]
                count_per_category_val[cur_category_name] += 1
            else:
                annotations_train["annotations"].append(annot)

        # Redraw random indices if counts aren't balanced per category
        counts_are_balanced = True
        for category_name in count_per_category.keys():
            num_total = count_per_category[category_name]
            num_val = count_per_category_val[category_name]
            current_is_balanced = (
                0.9 * args.ratio * num_total < num_val < 1.1 * args.ratio * num_total
            )
            if not current_is_balanced:
                counts_are_balanced = False
                break
        if not counts_are_balanced:
            continue

        # Add images
        for image in annotations["images"]:
            if image["id"] in val_indices:
                annotations_val["images"].append(image)
            else:
                annotations_train["images"].append(image)

        print("")
        print("Splits validation/train/total/ratio_val_total:")
        for category_name in count_per_category.keys():
            num_total = count_per_category[category_name]
            num_val = count_per_category_val[category_name]
            num_train = num_total - num_val
            print(
                f"{category_name} : {num_val} / {num_train} / {num_total} / {num_val / num_total}"
            )

        was_success = True
        break

    # Save
    with open(args.val, "wt") as f_out:
        json.dump(annotations_val, f_out)
    with open(args.train, "wt") as f_out:
        json.dump(annotations_train, f_out)


    if not was_success:
        print("Failed to split the dataset with given ratio.")
