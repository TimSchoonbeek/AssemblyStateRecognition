#!/bin/bash


# Shell script demonstrates how to make 5 executions with different seeds
# Settings for this script are best-performing: SupCon with ISIL for ViT-S

DATAPATH="path/to/train/data"
RUNPATH="path/to/directory/to/save"
TESTDATAPATH="path/to/test/data"

#the seeds used in the manuscript
SEEDS=(123 1234 12345 2345 345)

# GPU job script commands
module load anaconda3
source activate path/to/conda/environment


NAME="job_name"
for FOLD in 0 1 2 3 4
do
  FOLD_NAME="${NAME}_k${FOLD}"
  SEED=${SEEDS[$FOLD]}
  echo "Performing run with K-fold: ${FOLD}: training model ${FOLD_NAME}. Seed: ${SEED}"
  python train.py $FOLD_NAME \
  --data_path $DATAPATH \
  --run_path $RUNPATH \
  --seed $SEED \
  --n_real 8 \
  --n_synth 8 \
  --n_bg 64 \
  --exclude_bg \
  --img_h 224 \
  --img_w 224 \
  --model vit_small_patch16_224.augreg_in21k_ft_in1k

  wait

  RUN_DIR="${RUNPATH}/${FOLD_NAME}"
  python test.py $RUN_DIR --checkpoint best.pth --data_path $DATAPATH --h 224 --w 224
  wait
  python test_errors.py $RUN_DIR --checkpoint best.pth --data_path $TESTDATAPATH --img_h 224 --img_w 224
  wait
done
