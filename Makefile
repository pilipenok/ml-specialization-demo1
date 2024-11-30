PROJECT_ID ?= ml-spec-taxi
PROJECT_NUMBER ?= 591694990827
PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE ?= ./taxi.json
DATAFLOW_SYSTEM_FILES_BUCKET ?= taxi-dataflow-system
DATAFLOW_TEMP_FILES_BUCKET ?= taxi-dataflow-temp
FUNCTION_SYSTEM_FILES_BUCKET ?= taxi-function-system
OUTPUT_BUCKET ?= ml-spec-taxi-output
DATASET_NAME ?= chicago_taxi_trips
SOURCE_TABLE_NAME ?= taxi_trips
PATH_TO_TRIGGER_FUNCTION_SOURCE_ZIP_FILE ?= ./triggerfunction.zip
PATH_TO_CLEANUP_FUNCTION_SOURCE_ZIP_FILE ?= ./cleanupfunction.zip


gcp:
	# gcloud components update
	# gcloud auth login
	gcloud config set project ${PROJECT_ID}
	gcloud services enable cloudresourcemanager.googleapis.com
	gcloud services enable cloudbuild.googleapis.com
	gcloud services enable cloudscheduler.googleapis.com
	gcloud services enable dataflow.googleapis.com
	gcloud services enable bigquery.googleapis.com
	gcloud services enable iam.googleapis.com
	gcloud services enable compute.googleapis.com
	gcloud services enable cloudfunctions.googleapis.com
	gcloud services enable aiplatform.googleapis.com

gcp-res:
	gcloud storage buckets create gs://${DATAFLOW_SYSTEM_FILES_BUCKET}
	gcloud storage buckets create gs://${DATAFLOW_TEMP_FILES_BUCKET}
	gcloud storage buckets create gs://${FUNCTION_SYSTEM_FILES_BUCKET}
	gcloud storage buckets create gs://${OUTPUT_BUCKET}

tf-init:
	cd deployment && terraform init

tf-plan:
	cd deployment &&  terraform plan -var project=${PROJECT_ID} \
   -var service-account-key-location=${PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE} \
   -var dataflow-system-files-bucket=${DATAFLOW_SYSTEM_FILES_BUCKET} \
   -var dataflow-temp-files-bucket=${DATAFLOW_TEMP_FILES_BUCKET} \
   -var function-system-files-bucket=${FUNCTION_SYSTEM_FILES_BUCKET} \
   -var output-bucket=${OUTPUT_BUCKET} \
   -var dataset=${DATASET_NAME} \
   -var source-table=${SOURCE_TABLE_NAME} \
   -var trigger-function-location=${PATH_TO_TRIGGER_FUNCTION_SOURCE_ZIP_FILE} \
   -var cleanup-function-location=${PATH_TO_CLEANUP_FUNCTION_SOURCE_ZIP_FILE}

tf-apply:
	cd deployment &&  terraform apply -var project=${PROJECT_ID} \
   -var service-account-key-location=${PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE} \
   -var dataflow-system-files-bucket=${DATAFLOW_SYSTEM_FILES_BUCKET} \
   -var dataflow-temp-files-bucket=${DATAFLOW_TEMP_FILES_BUCKET} \
   -var function-system-files-bucket=${FUNCTION_SYSTEM_FILES_BUCKET} \
   -var output-bucket=${OUTPUT_BUCKET} \
   -var dataset=${DATASET_NAME} \
   -var source-table=${SOURCE_TABLE_NAME} \
   -var trigger-function-location=${PATH_TO_TRIGGER_FUNCTION_SOURCE_ZIP_FILE} \
   -var cleanup-function-location=${PATH_TO_CLEANUP_FUNCTION_SOURCE_ZIP_FILE}

