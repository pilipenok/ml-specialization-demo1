"""Constants for the taxi model.

These values can be tweaked to affect model training performance.
"""

HIDDEN_UNITS_DEEP_TANH = [32,16,8]
HIDDEN_UNITS_DEEP_RELU = [16,8,4]
HIDDEN_UNITS_MIX = [4,4,2]
HIDDEN_UNITS_WIDE = [128,16,2]
HIDDEN_UNITS = [4,1]

LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40
