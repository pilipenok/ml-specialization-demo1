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
"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text
from tfx.orchestration import pipeline

from ml_metadata.proto import metadata_store_pb2

from pipeline.components import example_gen, statistics_gen, schema_gen, example_validator, transform, trainer, model_resolver, evaluator, pusher


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> pipeline.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""

    _example_gen = example_gen()
    _statistics_gen = statistics_gen(
        example_gen=_example_gen
    )
    _schema_gen = schema_gen(
        statistics_gen=_statistics_gen
    )
    _example_validator = example_validator(
        statistics_gen=_statistics_gen, 
        schema_gen=_schema_gen
    )
    _trainer = trainer(
        example_gen=_example_gen, 
        schema_gen=_schema_gen
    )
    _model_resolver = model_resolver(),
    _evaluator = evaluator(
        example_gen=_example_gen,
        trainer=_trainer,
        model_resolver=_model_resolver
    )
    _pusher = pusher(
        trainer=_trainer
    )

    components = [
        _example_gen,
        _statistics_gen,
        _schema_gen,
        _example_validator,
        #_transform,
        _trainer,
        _model_resolver,
        _evaluator,
        _pusher
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        # Change this value to control caching of execution results. Default value
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
    )
