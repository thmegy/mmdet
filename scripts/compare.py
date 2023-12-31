"""
Compares inference results with a set of ground-truth, human-made annotations.

WARNING: this script needs the annotation_metrics package in order to work. You can install it 
from the requirements.txt file or by running the command
`pip install git+ssh://git@Gitlab.Logiroad.com:2316/v.papelard/annotations-comparator.git`

Arguments:
- inferences: Path to the JSON file that contains the inferences in JSON format,
as generated by this code: 
https://gitlab.logiroad.com/theom/ai-vs-videocoding/blob/master/scripts/utils/videocoding.py
- ref: path to a folder that contains the JSON annotations you want to use as a ground-truth
(in L2R fomat).
- geo: path to the geolocation CSV file.
- distance_threshold: Threshold in meters below which two reported degradations are deemed to be the same. Defaults to 10 meters if no value was given
- score_thresholds: Path to a list of score thresholds for each degradation class in a JSON file.

This script will display the following metrics:
- precision
- recall
- F1 score
- truepos_precision: the weighted number of true positives used when computing the precision
- truepos_recall: the weighted number of true positives used when computing the recall
- falsepos: the weighted number of false positives used when computing the precision
- falseneg: the weighted number of false negatives used when computing the recall
"""

DEFAULT_SCORE_THRESHOLDS = 8 * [0.5]
DEFAULT_DISTANCE_THRESHOLD = 10
CLASSES_INDICES = {
    "Arrachement_pelade": 0,
    "Faiencage": 1,
    "Nid_de_poule": 2,
    "Transversale": 3,
    "Longitudinale": 4,
    "Remblaiement_de_tranchees": 5,
    "Pontage": 6,
    "Autre_reparation": 7,
}


import argparse
import json
import os

import pandas as pd

from annotation_metrics.comparator import Comparator


def parse_arguments() -> argparse.Namespace:
    """
    Parse the arguments and returns them.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inferences",
        required=True,
        help="Path to the file that contains the inferences in a JSON format",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Path to the folder that contains the ground-truth annotations in JSON",
    )
    parser.add_argument(
        "--geo",
        required=True,
        help="Path to the CSV file that contains the geolocation data for the video",
    )
    parser.add_argument(
        "--distance_threshold",
        required=False,
        help="Threshold in meters below which two reported degradations are deemed to be the same. Defaults to 10 meters if no value was given",
    )
    parser.add_argument(
        "--score_thresholds",
        required=False,
        help="""Filepath to a JSON file that contains the score threshold for each degradation class. This JSON must follow this structure: \n{\n
            "Arrachement_pelade": 0.4,\n
            "Faiencage": 0.2,\n
            "Nid_de_poule": 0.76,\n
            "Transversale": 0.12,\n
            "Longitudinale": 0.34,\n
            "Remblaiement_de_tranchees": 0.5,\n
            "Pontage": 0.6,\n
            "Autre_reparation": 0.7,\n
            }.\n
            If not provided, all thresholds are set at 0.5.
            """,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    comparator = Comparator()
    annot_list = []
    score_thresholds = []

    # Loading AI inferences
    with open(args.inferences, "r") as inference_file:
        annot_list.append(json.load(inference_file))

    # Loading base truth videocodings
    files_list = os.listdir(args.ref)
    for file in files_list:
        if file.endswith("json"):
            with open(args.ref + "/" + file, "r") as l2r_annotations:
                annot_list.append(json.load(l2r_annotations))

    # Loading geolocation
    geolocation = pd.read_csv(args.geo, sep=";")

    # Loading score thresholds
    if not args.score_thresholds:
        score_thresholds = DEFAULT_SCORE_THRESHOLDS
    else:
        with open(args.score_thresholds, "r") as file:
            thresholds = json.load(file)
            score_thresholds = len(CLASSES_INDICES) * [0.5]

            for degradation_class in CLASSES_INDICES:
                score_thresholds[CLASSES_INDICES[degradation_class]] = float(
                    thresholds[degradation_class]
                )

    # Loading distance threshold
    if not args.distance_threshold:
        distance_threshold = DEFAULT_DISTANCE_THRESHOLD
    else:
        distance_threshold = float(args.distance_threshold)

    # Performing the comparison
    comparison_results = comparator.compare(
        annots=annot_list,
        geoptis=geolocation,
        classes="all",
        score_thresholds=score_thresholds,
        threshold_dist=distance_threshold,
    )

    print("COMPARISON RESULTS FOR EACH DEGRADATION CLASS:")
    for class_name in comparison_results[0]:
        print(f"{class_name}:")
        for key in comparison_results[0][class_name]:
            print(f"\t{key}: {comparison_results[0][class_name][key]}")
        print()
