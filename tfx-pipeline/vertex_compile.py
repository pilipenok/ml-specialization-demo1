from tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner import KubeflowV2DagRunnerConfig, KubeflowV2DagRunner
from google.cloud import storage

from absl import logging

from pipeline import configs
from pipeline import pipeline

PIPELINE_NAME = configs.PIPELINE_NAME
PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'
GS_PIPELINE_DEFINITION_URI = f'gs://{configs.GCS_BUCKET_NAME}/{PIPELINE_DEFINITION_FILE}'
MODEL_PATH = '../models/keras/baseline_advanced/model.py'
GS_MODEL_URI = configs.MODULE_FILE


def main():
    print(f'Compiling to {PIPELINE_DEFINITION_FILE}')
    runner_config = KubeflowV2DagRunnerConfig(
        display_name='tfx-vertex-pipeline-{}'.format(PIPELINE_NAME),
        default_image='us.gcr.io/or2--epm-gcp-by-meetup2-t1iylu/taxi-pipeline-vertex',
    )

    KubeflowV2DagRunner(
        config=runner_config,
        output_filename=PIPELINE_DEFINITION_FILE
    ).run(
        pipeline.create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=configs.PIPELINE_ROOT,
            enable_vertex=True,
            enable_transform=False,
            enable_hyperparameters_tuning=False
        )
    )

    print(f'Uploading pipeline to {GS_PIPELINE_DEFINITION_URI}')
    client=storage.Client()
    storage.Blob.from_string(GS_PIPELINE_DEFINITION_URI, client=client).upload_from_filename(PIPELINE_DEFINITION_FILE)
    print(f'Uploading model to {GS_MODEL_URI}')
    storage.Blob.from_string(GS_MODEL_URI, client=client).upload_from_filename(MODEL_PATH)
    print('Done.')


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main()
