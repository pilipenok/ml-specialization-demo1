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
"""Define LocalDagRunner to run the pipeline locally."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from pipeline import configs
from pipeline import pipeline


METADATA_PATH = os.path.join(
    configs.LOCAL_OUTPUT_DIR,
    'tfx_metadata',
    configs.PIPELINE_NAME,
    'metadata.db'
)


def run():
    """Define a local pipeline."""

    LocalDagRunner().run(
        pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=configs.LOCAL_PIPELINE_ROOT,
            metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
            enable_vertex=False,
            enable_transform=False,
            enable_hyperparameters_tuning=True
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
