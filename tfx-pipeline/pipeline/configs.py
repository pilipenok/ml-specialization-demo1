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
"""TFX taxi template configurations.

This file defines environments for a TFX taxi pipeline.
"""

import os
import tfx

PIPELINE_NAME = 'tfx-pipeline-ml'
GOOGLE_CLOUD_PROJECT = 'o-epm-gcp-by-meetup1-ml-t1iylu'
GCS_BUCKET_NAME = 'chicago-taxi-ml-demo-1'
GOOGLE_CLOUD_REGION = 'us-central1'

# Specifies data file directory. DATA_PATH should be a directory containing CSV files for CsvExampleGen in this example. 
DATA_PATH = f'gs://{GCS_BUCKET_NAME}/trips/bucket_target/'
LOCAL_DATA_PATH = '.' # local path to 'trips_small.csv'

# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.keras.baseline_advanced.model.run_fn'
MODULE_FILE = 'gs://chicago-taxi-ml-demo-1/model.py'
SERVING_MODEL_DIR = 'gs://chicago-taxi-ml-demo-1/serving_model'

TRAIN_NUM_STEPS = 10#0000
EVAL_NUM_STEPS = 10#00
EVAL_ACCURACY_THRESHOLD = 0.6
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

USE_GPU = False
# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
GCP_AI_PLATFORM_TRAINING_ARGS = {
    'project': GOOGLE_CLOUD_PROJECT,
    'region': GOOGLE_CLOUD_REGION,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use
    # a public container image matching the installed version of TFX.
    'masterConfig': {
      'imageUri': PIPELINE_IMAGE
    },
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
GCP_AI_PLATFORM_SERVING_ARGS = {
    'model_name': PIPELINE_NAME.replace('-','_'),  # '-' is not allowed.
    'project_id': GOOGLE_CLOUD_PROJECT,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [GOOGLE_CLOUD_REGION],
    'endpoint_name': 'chicago_taxi_model_endoint',
    'min_replica_count': 1,
    'max_replica_count': 2,
    'machine_type': 'n1-standard-2'
}
# If you are looking for the url to query the Endpoint,
# that's in a property pushed_destination of the pushed_model output artifact:
# model_pushed_artifact = pusher.outputs[PUSHED_MODEL_KEY]
# pushed_destination = model_pushed_artifact.get_string_custom_property("pushed_destination")
# Together with the VERTEX_REGION_KEY, you can create the url with something like:
# f"https://{region}-aiplatform.googleapis.com/v1/{pushed_destination}:predict".


# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = f'gs://{GCS_BUCKET_NAME}'
LOCAL_OUTPUT_DIR = './'

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = f'{OUTPUT_DIR}/tfx_pipeline_output/{PIPELINE_NAME}'
LOCAL_PIPELINE_ROOT = f'{LOCAL_OUTPUT_DIR}/tfx_pipeline_output/{PIPELINE_NAME}'

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = f'{PIPELINE_ROOT}/serving_model'
LOCAL_SERVING_MODEL_DIR = f'{LOCAL_PIPELINE_ROOT}/serving_model'

# NEW: Configuration for Vertex AI Training.
# This dictionary will be passed as `CustomJobSpec`.
GCP_VERTEX_AI_TRAINING_ARGS = {
    'project': GOOGLE_CLOUD_PROJECT,
    'region': GOOGLE_CLOUD_REGION,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use
    # a public container image matching the installed version of TFX.
    #'masterConfig': {
    #  'imageUri': PIPELINE_IMAGE
    #},
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)

    'worker_pool_specs': [{
        'machine_spec': {'machine_type': 'n1-standard-4',},
        'replica_count': 1,
        'container_spec': {
            'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
        },
    }],
}

if USE_GPU:
    # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
    # for available machine types.
    GCP_VERTEX_AI_TRAINING_ARGS['worker_pool_specs'][0]['machine_spec'].update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })