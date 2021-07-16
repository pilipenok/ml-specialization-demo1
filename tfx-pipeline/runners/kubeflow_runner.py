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

from pipelines.pipeline import create_pipeline, configs

from tfx.orchestration.kubeflow import kubeflow_dag_runner as runner
from tfx.utils import telemetry_utils


def run():
    """Define a kubeflow pipeline."""

    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    # If you use Kubeflow, metadata will be written to MySQL database inside
    # Kubeflow cluster.
    metadata_config = runner.get_default_kubeflow_metadata_config()

    runner_config = runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        tfx_image=configs.PIPELINE_IMAGE
    )
    pod_labels = runner.get_default_pod_labels()
    pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: 'tfx-template'})

    pipeline = create_pipeline(
        pipeline_name=configs.PIPELINE_NAME,
        pipeline_root=configs.PIPELINE_ROOT,
    )

    runner.KubeflowDagRunner(
        config=runner_config,
        pod_labels_to_attach=pod_labels
    ).run(
        pipeline=pipeline
    )


if __name__ == '__main__':
    from absl import logging
    logging.set_verbosity(logging.INFO)
    run()
