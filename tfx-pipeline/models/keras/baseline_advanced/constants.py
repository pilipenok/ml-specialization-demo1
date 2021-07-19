"""Constants for the taxi model.

These values can be tweaked to affect model training performance.
"""

HIDDEN_UNITS = [16, 8]
HIDDEN_UNITS_ADVANCED = [16,8,8,4]
HIDDEN_UNITS_ADVANCED2 = [4,4,2]
HIDDEN_UNITS_ADVANCED_SINK = [64,8,2]

LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40
