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
"""Define KubeflowDagRunner to run the pipeline using Kubeflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils


def run():
    """Define a kubeflow pipeline."""

    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    # If you use Kubeflow, metadata will be written to MySQL database inside
    # Kubeflow cluster.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        tfx_image=configs.PIPELINE_IMAGE
    )
    pod_labels = kubeflow_dag_runner.get_default_pod_labels()
    pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: 'tfx-template'})
    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config, 
        pod_labels_to_attach=pod_labels
    ).run(
        pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=configs.PIPELINE_ROOT,
            data_path=configs.DATA_PATH,
            preprocessing_fn=configs.PREPROCESSING_FN,
            run_fn=configs.RUN_FN,
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
