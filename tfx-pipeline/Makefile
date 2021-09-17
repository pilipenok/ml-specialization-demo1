GOOGLE_CLOUD_PROJECT ?= or2--epm-gcp-by-meetup2-t1iylu
ENDPOINT ?= https://4fe2a1125fd9595a-dot-us-central1.pipelines.googleusercontent.com
PIPELINE_NAME ?= tfx-pipeline-ml


create:
	tfx pipeline create --pipeline-path=kubeflow_runner.py --endpoint=${ENDPOINT} --build-image

update:
	tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint=${ENDPOINT} --build-image
    
run:
	tfx run create --pipeline-name ${PIPELINE_NAME} --endpoint=${ENDPOINT}

create-run: create run
update-run: update run
