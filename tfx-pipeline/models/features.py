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

import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_transform as tft

from pipeline import components


FEATURE_KEYS = 'area,is_holiday,day_of_week,year,month,day,hour24,hour12,day_period,avg_total_per_trip_prev4h_area,avg_total_per_trip_prev4h_city,avg_ntrips_prev_4h_area,avg_ntrips_prev_4h_city'.split(',')


LABEL_KEY = 'relative_demand'


# Since we're not generating or creating a schema, we will instead create a feature spec.
FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
        for feature in 'avg_total_per_trip_prev4h_area,avg_total_per_trip_prev4h_city,avg_ntrips_prev_4h_area,avg_ntrips_prev_4h_city'.split(',')
    },
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
        for feature in 'area,day_of_week,year,month,day,hour24,hour12'.split(',')
    },
    'is_holiday': tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    'day_period': tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
}


def get_schema():
    #return schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    return components.schema_gen().outputs['schema']


