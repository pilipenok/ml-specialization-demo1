"""TFX taxi model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List

import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_transform as tft

from pipeline import components


# Keys
from models.keras.baseline_advanced.constants import LABEL_KEY
# LABEL_KEY = 'n_trips' # 'log_n_trips' # 

FEATURE_KEYS = \
   "area,year,"\
   "quarter,quarter_num,quarter_cos,quarter_sin,"\
   "month,month_num,month_cos,month_sin,"\
   "day,day_num,day_cos,day_sin,"\
   "hour,hour_num,hour_cos,hour_sin,"\
   "day_period,"\
   "week,week_num,week_cos,week_sin,"\
   "day_of_week,day_of_week_num,day_of_week_cos,day_of_week_sin,"\
   "weekday_hour_num,weekday_hour_cos,weekday_hour_sin,"\
   "yearday_hour_num,yearday_hour_cos,yearday_hour_sin,"\
   "is_weekend,is_holiday"\
    .split(',')

# At least one feature is needed.

# Name of features which have continuous float values. These features will be
# used as their own values.
DENSE_FLOAT_FEATURE_KEYS = \
   "year,"\
   "quarter_num,quarter_cos,quarter_sin,"\
   "month_num,month_cos,month_sin,"\
   "day_num,day_cos,day_sin,"\
   "hour_num,hour_cos,hour_sin,"\
   "week_num,week_cos,week_sin,"\
   "day_of_week_num,day_of_week_cos,day_of_week_sin,"\
   "weekday_hour_num,weekday_hour_cos,weekday_hour_sin,"\
   "yearday_hour_num,yearday_hour_cos,yearday_hour_sin"\
    .split(',')

# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
BUCKET_FEATURE_KEYS = "area,quarter,month,day,hour,week,day_of_week".split(',')
# Number of buckets used by tf.transform for encoding each feature. The length
# of this list should be the same with BUCKET_FEATURE_KEYS.
BUCKET_FEATURE_BUCKET_COUNT = [77,4,12,31,12,53,7]

# Name of features which have categorical values which are mapped to integers.
# These features will be used as categorical features.
CATEGORICAL_FEATURE_KEYS = "day_period,is_weekend,is_holiday".split(',')
# Number of buckets to use integer numbers as categorical features. The length
# of this list should be the same with CATEGORICAL_FEATURE_KEYS.
CATEGORICAL_FEATURE_MAX_VALUES = [2,2,2]

# Name of features which have string values and are mapped to integers.
VOCAB_FEATURE_KEYS = []
# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 2
# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10


def transformed_name(key: Text) -> Text:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'


def vocabulary_name(key: Text) -> Text:
  """Generate the name of the vocabulary feature from original name."""
  return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
  """Transform multiple feature names at once."""
  return [transformed_name(key) for key in keys]


# Since we're not generating or creating a schema, we will instead create a feature spec.
FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
        for feature in DENSE_FLOAT_FEATURE_KEYS
    },
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
        for feature in BUCKET_FEATURE_KEYS
    },
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.string)
        for feature in CATEGORICAL_FEATURE_KEYS
    },
    LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32 if '_num' in LABEL_KEY else tf.int64),
#     LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
}
FEATURE_SPEC['year'] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)


def get_schema():
    return schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    #return components.schema_gen().outputs['schema']


