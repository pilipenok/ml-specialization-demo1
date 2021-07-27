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
"""TFX taxi preprocessing.

This file defines a template for TFX Transform component.
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_transform as tft

from models import features
from absl import logging


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
        inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
        Map from string feature key to transformed feature operations.
    """
    
    outputs = dict(
        avg_total_per_trip_prev4h_area=tft.scale_to_z_score(_fill_in_missing(inputs['avg_total_per_trip_prev4h_area'])),
        avg_total_per_trip_prev4h_city=tft.scale_to_z_score(_fill_in_missing(inputs['avg_total_per_trip_prev4h_city'])),
        avg_ntrips_prev_4h_area=tft.scale_to_z_score(_fill_in_missing(inputs['avg_ntrips_prev_4h_area'])),
        avg_ntrips_prev_4h_city=tft.scale_to_z_score(_fill_in_missing(inputs['avg_ntrips_prev_4h_city'])),
        hour24=tft.bucketize(_fill_in_missing(inputs['hour24']), 4),
        area=tft.bucketize(_fill_in_missing(inputs['area']), 77),
        is_holiday=_fill_in_missing(inputs['is_holiday']),
        day_of_week=_fill_in_missing(inputs['day_of_week']),
        month=_fill_in_missing(inputs['month']),
        day=_fill_in_missing(inputs['day']),
        hour12=_fill_in_missing(inputs['hour12']),
        day_period=_fill_in_missing(inputs['day_period']),
        relative_demand=inputs["relative_demand"]
    )

    return outputs
