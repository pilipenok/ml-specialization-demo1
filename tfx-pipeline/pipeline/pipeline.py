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

import pipeline.components as pc

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    enable_vertex=False,
    enable_transform=False,
    enable_hyperparameters_tuning=False
) -> pipeline.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""

    _example_gen = pc.example_gen()
    examples = _example_gen.outputs['examples']

    _statistics_gen = pc.statistics_gen(examples=examples)
    statistics = _statistics_gen.outputs['statistics']

    _schema_gen = pc.schema_gen(statistics=statistics,)
    schema = _schema_gen.outputs['schema']

    _transform = None
    transform_graph = None
    if enable_transform:
        _transform = pc.transform(examples=examples, schema=schema)
        examples = _transform.outputs['transformed_examples']
        transform_graph = _transform.outputs['transform_graph']

    _example_validator = pc.example_validator(statistics=statistics, schema=schema)

    _tuner = None
    #TODO: get saved params if tuning is off
    hyperparameters = None
    if enable_hyperparameters_tuning:
        _tuner = pc.tuner(schema=schema, examples=examples, transform_graph=transform_graph)
        hyperparameters = _tuner.outputs['best_hyperparameters']

    trainer = pc.trainer_vertex if enable_vertex else pc.trainer
    _trainer = trainer(schema=schema, examples=examples, transform_graph=transform_graph, hyperparameters=hyperparameters)
    model = _trainer.outputs['model']

    _model_resolver = pc.model_resolver()
    baseline_model = _model_resolver.outputs['model']

    _evaluator = pc.evaluator(examples=examples, model=model, baseline_model=baseline_model)
    model_blessing = _evaluator.outputs['blessing']

    pusher = pc.pusher_vertex if enable_vertex else pc.pusher
    _pusher = pusher(
        model=model,
        model_blessing=model_blessing
    )

    components = [
        _example_gen,
        _statistics_gen,
        _schema_gen,
        _example_validator,
        _trainer,
        _model_resolver,
        _evaluator,
        _pusher
    ]

    if _transform:
        components.append(_transform)
    if _tuner:
        components.append(_tuner)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        # Change this value to control caching of execution results. Default value
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
    )
