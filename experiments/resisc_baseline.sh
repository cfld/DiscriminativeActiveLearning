#!/usr/bin/env bash

declare -i EXPERIMENT_IDX=0
declare -i BATCH_SIZE=50
declare -i INITIAL_SIZE=100
declare -i ITERATIONS=15
declare -a StringArray=("CoreSetMIP" "Discriminative" "DiscriminativeLearned" \
                        "DiscriminativeAE" "DiscriminativeStochastic" "Uncertainty" "Bayesian" \
                        "UncertaintyEntropy" "BayesianEntropy" "EGL" "Adversarial")


for val in "${StringArray[@]}"; do
    python /home/ebarnett/DiscriminativeActiveLearning/main.py \
                   --experiment_index $EXPERIMENT_IDX --data_type resisc_features \
                   --batch_size $BATCH_SIZE --initial_size $INITIAL_SIZE --iterations $ITERATIONS \
                   --method $val \
                   --experiment_folder /home/ebarnett/DiscriminativeActiveLearning/experiments/ \
                   --gpu 1
done