"""Constants for the taxi model.

These values can be tweaked to affect model training performance.
"""

from features import LABEL_KEY


HIDDEN_UNITS_DEEP_TANH = [64,32,16]
HIDDEN_UNITS_DEEP_RELU = [16,8,4]
HIDDEN_UNITS_WIDE = [512,64,4]
HIDDEN_UNITS_MIX = [128,32,2]
HIDDEN_UNITS = [4,1]

EPOCHS = 10

LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16


MODEL_NAME = LABEL_KEY+"-"+EPOCHS+"-"+TRAIN_BATCH_SIZE
