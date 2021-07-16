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
"""Define KubeflowV2DagRunner to run the pipeline."""

import os

from pipelines.pipeline import create_pipeline, configs

from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.tools.cli import labels

PIPELINE_NAME = configs.PIPELINE_NAME + '-v2'
PIPELINE_ROOT = configs.PIPELINE_ROOT


def run():
    """Define a pipeline to be executed using Kubeflow V2 runner."""
    # TODO(b/157598477) Find a better way to pass parameters from CLI handler to
    # pipeline DSL file, instead of using environment vars.
    tfx_image = os.environ.get(labels.TFX_IMAGE_ENV)
    project_id = os.environ.get(labels.GCP_PROJECT_ID_ENV)
    api_key = os.environ.get(labels.API_KEY_ENV)

    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        project_id=project_id,
        display_name=f'tfx-kubeflow-v2-pipeline-{PIPELINE_NAME}',
        default_image=tfx_image
    )

    pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT
    )

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config)

    if os.environ.get(labels.RUN_FLAG_ENV, False):
        # Only trigger the execution when invoked by 'run' command.
        runner.run(pipeline=pipeline, api_key=api_key)
    else:
        runner.compile(pipeline=pipeline, write_out=True)


if __name__ == '__main__':
    from absl import logging
    logging.set_verbosity(logging.INFO)
    run()
