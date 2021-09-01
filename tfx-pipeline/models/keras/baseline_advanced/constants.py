"""Constants for the taxi model.

These values can be tweaked to affect model training performance.
"""


baseline = True # False # 
task = 'regr' # 'class' # 

regularizer = False # True #
dropout = False # True #

NUM_CLASSES = 11
EPOCHS = 25
TRAIN_BATCH_SIZE = 16
TRAIN_NUM_STEPS = 50000
EVAL_BATCH_SIZE = 16
EVAL_NUM_STEPS = 1000
ES_PATIENCE = 0 if baseline else 3


LABEL_KEY = 'trips_bucket' if task == 'class' else 'trips_bucket_num' # 'log_n_trips' # 'n_trips' #

MODEL_NAME = f"{LABEL_KEY}-"\
             f"{'baseline' if baseline else 'advanced'}"\
             f"{'_regul' if regularizer and not baseline else ''}"\
             f"{'_drop' if dropout and not baseline else ''}"\
             f"-{EPOCHS}-{TRAIN_BATCH_SIZE}"


LEARNING_RATE = 0.001

HIDDEN_UNITS_BASE_DEEP = [32,16,8]
HIDDEN_UNITS_BASE_CONCAT = [32, NUM_CLASSES if task=='class' else 1]

HIDDEN_UNITS_ADV_DEEP = [64,32,16]
HIDDEN_UNITS_ADV_EMBED = [8]
HIDDEN_UNITS_ADV_MIX = [16,4]
HIDDEN_UNITS_ADV_WIDE = [512,64,4]
HIDDEN_UNITS_ADV_CONCAT = [4, NUM_CLASSES if task=='class' else 1]
