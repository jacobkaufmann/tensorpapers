#!/bin/bash

# base-dir: indicates where the ILSVRC folder from Kaggle resides
# data-dir: indicates where you would like the preprocessed data to reside
# tf-slim-dir: indicates where cloned Tensorflow slim directory is located
# usage: ./imagenet.sh [base-dir] [data-dir] [tf-slim-dir]

set -e

if [[ $# -ne 3 ]]; then
    echo "illegal number of parameters"
    echo "usage download_and_convert_imagenet.sh [base-dir] [data-dir] [tf-slim-dir]"
    exit
fi

# Note base directory and create output directory
BASE_DIR="${1}"
DATA_DIR="${2}"
TF_SLIM_DIR="${3}"
mkdir -p "${DATA_DIR}"

# Note locations for train and validation directories
IMAGES_DIR="${BASE_DIR}/Data/CLS-LOC"
TRAIN_DIR="${IMAGES_DIR}/train/"
VAL_DIR="${IMAGES_DIR}/val/"

# Preprocess validation data
echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="${TF_SLIM_DIR}/datasets/preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="${TF_SLIM_DIR}/datasets/imagenet_2012_validation_synset_labels.txt"
"${PREPROCESS_VAL_SCRIPT}" "${VAL_DIR}" "${VAL_LABELS_FILE}"

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
LABELS_FILE="${TF_SLIM_DIR}/datasets/imagenet_lsvrc_2015_synsets.txt"
BOUNDING_BOX_SCRIPT="${TF_SLIM_DIR}/datasets/process_bounding_boxes.py"
BOUNDING_BOX_FILE="${DATA_DIR}/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${BASE_DIR}/Annotations/CLS-LOC/train/"
"${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
 | sort >"${BOUNDING_BOX_FILE}"
echo "Finished preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="${TF_SLIM_DIR}/datasets/build_imagenet_data.py"
OUTPUT_DIR="${DATA_DIR}"
IMAGENET_METADATA_FILE="${TF_SLIM_DIR}/datasets/imagenet_metadata.txt"

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VAL_DIR}" \
  --output_directory="${OUTPUT_DIR}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"
  echo "Finished building TFRecords for ImageNet data."