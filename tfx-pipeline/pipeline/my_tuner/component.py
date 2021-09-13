# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Example of a Hello World TFX custom component.

This custom component simply reads tf.Examples from input and passes through as
output.  This is meant to serve as a kind of starting point example for creating
custom components.

This component along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""

from tfx.dsl.components.base import executor_spec
from tfx.examples.custom_components.hello_world.hello_component import executor
from tfx.components.tuner import component as tuner_component


class MyTunerComponentSpec(tuner_component.Tuner):
    """TFX component for model hyperparameter tuning on AI Platform Training."""

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)
