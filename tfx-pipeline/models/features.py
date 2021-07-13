# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX taxi model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List

# At least one feature is needed.

# Name of features which have continuous float values. These features will be
# used as their own values.
DENSE_FLOAT_FEATURE_KEYS = ['avg_fare_per_trip']

# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
BUCKET_FEATURE_KEYS = []
# Number of buckets used by tf.transform for encoding each feature. The length
# of this list should be the same with BUCKET_FEATURE_KEYS.
BUCKET_FEATURE_BUCKET_COUNT = []

# Name of features which have categorical values which are mapped to integers.
# These features will be used as categorical features.
CATEGORICAL_FEATURE_KEYS = ['pickup_community_area', 'day_of_week', 'month', 'hour_of_day']
# Number of buckets to use integer numbers as categorical features. The length
# of this list should be the same with CATEGORICAL_FEATURE_KEYS.
CATEGORICAL_FEATURE_MAX_VALUES = [100, 7, 2, 12, 24, 2]

# Name of features which have string values and are mapped to integers.
VOCAB_FEATURE_KEYS = ['am_pm','is_us_holiday']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 2

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

# Keys
LABEL_KEY = 'number_of_trips'


def transformed_name(key: Text) -> Text:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'


def vocabulary_name(key: Text) -> Text:
  """Generate the name of the vocabulary feature from original name."""
  return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
  """Transform multiple feature names at once."""
  return [transformed_name(key) for key in keys]
