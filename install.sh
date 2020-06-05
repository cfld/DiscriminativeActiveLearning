#!/usr/bin/env bash

conda create --name active_learning_env python=3.7
conda activate active_learning_env

conda install -y keras
conda install -y scikit-learn
pip install cleverhans
pip install tensorflow==1.13.2
