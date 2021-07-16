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

from pipelines.pipeline import create_pipeline, configs

from tfx.orchestration import metadata
from tfx.orchestration.local import local_dag_runner  as runner

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
OUTPUT_DIR = '..'

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.
PIPELINE_NAME = configs.PIPELINE_NAME + '-local'
PIPELINE_ROOT = f"{OUTPUT_DIR}/tfx_pipeline_output/{PIPELINE_NAME}"
METADATA_PATH = f"{OUTPUT_DIR}/tfx_metadata/{PIPELINE_NAME}/metadata.db"


def run():
    """Define a local pipeline."""
    pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
    )

    runner.LocalDagRunner().run(pipeline=pipeline)


if __name__ == '__main__':
    from absl import logging
    logging.set_verbosity(logging.INFO)
    run()
