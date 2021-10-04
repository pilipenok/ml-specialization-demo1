from kfp.v2.google import client
from pipeline import configs

PIPELINE_NAME = configs.PIPELINE_NAME
PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'
GS_PIPELINE_DEFINITION_URI = f'gs://{configs.GCS_BUCKET_NAME}/{PIPELINE_DEFINITION_FILE}'


def main():
    client.AIPlatformClient(
        project_id=configs.GOOGLE_CLOUD_PROJECT,
        region=configs.GOOGLE_CLOUD_REGION
    ).create_run_from_job_spec(
        GS_PIPELINE_DEFINITION_URI,
        enable_caching=False
    )


if __name__ == '__main__':
    main()
