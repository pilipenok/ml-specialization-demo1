GOOGLE_CLOUD_PROJECT ?= o-epm-gcp-by-meetup1-ml-t1iylu
ENDPOINT ?= https://4e0115c7a3a887cd-dot-us-central1.pipelines.googleusercontent.com
PIPELINE_NAME ?= tfx-pipeline



update:
	tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint=${ENDPOINT} --build-image

create:
	tfx pipeline create --pipeline-path=kubeflow_runner.py --endpoint=${ENDPOINT} --build-image
    
run:
	tfx run create --pipeline-name ${PIPELINE_NAME} --endpoint=${ENDPOINT}
    
 
update-run: update run
