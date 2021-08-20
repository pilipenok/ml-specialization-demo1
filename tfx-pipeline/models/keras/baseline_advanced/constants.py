"""Constants for the taxi model.

These values can be tweaked to affect model training performance.
"""

baseline = True # False # 
LABEL_KEY = 'n_trips' # 'log_n_trips' # 

EPOCHS = 25
TRAIN_BATCH_SIZE = 16
TRAIN_NUM_STEPS = 50000
EVAL_BATCH_SIZE = 16
EVAL_NUM_STEPS = 1000
ES_PATIENCE = 3


model_name = 'baseline' if baseline else 'advanced'
MODEL_NAME = f"{LABEL_KEY}-{model_name}-{EPOCHS}-{TRAIN_BATCH_SIZE}"


LEARNING_RATE = 0.001

HIDDEN_UNITS_DEEP_TANH = [64,32,16]
HIDDEN_UNITS_DEEP_RELU = [16,8,4]
HIDDEN_UNITS_WIDE = [512,64,4]
HIDDEN_UNITS_MIX = [128,32,2]
HIDDEN_UNITS_CONCAT = [4,1]
