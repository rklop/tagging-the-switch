#!/usr/bin/env python
# config.py

import os
from transformers import (
RemBertConfig,
XLMRobertaConfig,
BertConfig,
DebertaV2Config
)

BASE_DIR = os.getcwd() 

# Dataset directory
DATASETS_DIR = os.path.join(BASE_DIR, 'data')

# Cleaned datasets directory
POS_TRAIN_FILE = os.path.join(DATASETS_DIR, 'POS', 'train.conll')
POS_DEV_FILE = os.path.join(DATASETS_DIR, 'POS', 'dev.conll')
POS_TEST_FILE = os.path.join(DATASETS_DIR, 'POS', 'test.conll')

LID_TRAIN_FILE = os.path.join(DATASETS_DIR, 'LID', 'train.conll')
LID_DEV_FILE = os.path.join(DATASETS_DIR, 'LID', 'dev.conll')
LID_TEST_FILE = os.path.join(DATASETS_DIR, 'LID', 'test.conll') 

"""
MODEL_NAMES | MODEL_CONFIGS

RemBERT: "google/rembert" | RemBertConfig
XLM-R: "xlm-roberta-base" or "xlm-roberta-large" | XLMRobertaConfig
mBERT: "bert-base-multilingual-cased" | BertConfig
"""

# Choose model name and config
MODEL_NAME ="xlm-roberta-base"
MODEL_CONFIG = RemBertConfig
SAVES_DIR = os.path.join(BASE_DIR, 'saves')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'model')

# Model save paths
SINGLE_TASK_POS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'single_task_pos-final')
SINGLE_TASK_LID_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'single_task_lid-final')
SEQUENTIAL_POS_LID_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'sequential_pos_lid-final')
SEQUENTIAL_LID_POS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'sequential_lid_pos-final')
MULTITASK_UNWEIGHTED_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'multitask_unweighted-final')
MULTITASK_WEIGHTED_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'multitask_weighted-final')

# Training Hyperparameters
LEARNING_RATE = 5e-6 
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 32 
PER_DEVICE_EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
MAX_SEQUENCE_LENGTH = 128

# Evaluation Metrics
METRIC_FOR_BEST_MODEL_SINGLE_TASK = "eval_f1" # Same for sequential and single-task
METRIC_FOR_BEST_MODEL_MULTITASK = "eval_combined_f1" 

# For weighted multitask training. 
LID_WEIGHT = 1.0 # default weight for LID task
POS_WEIGHT = 1.0 # default weight for POS task
UNCERTAINTY_WEIGHTING = False # Set to True for uncertainty-based weighting (Kendall et al., 2018)