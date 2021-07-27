GOOGLE_CLOUD_PROJECT ?= o-epm-gcp-by-meetup1-ml-t1iylu
ENDPOINT ?= https://38fa5fd7828860d4-dot-us-central1.pipelines.googleusercontent.com
PIPELINE_NAME ?= tfx-pipeline



build:
	tfx pipeline update --pipeline-path=kubeflow_runner.py --endpoint=${ENDPOINT} --build-image
    
run:
	tfx run create --pipeline-name ${PIPELINE_NAME} --endpoint=${ENDPOINT}
    
 
build-run: build run
