# Cloud environment setup (data pipeline part)
To set up Cloud environment we will use Terraform configuration.

1. In GCP project enable the following APIs:  
- Cloud Resource Manager API
- Cloud Build API
- Cloud Scheduler API
- Dataflow API
- BigQuery API
- Identity and Access Management (IAM) API
- Compute Engine API
- Cloud Functions API

2. Prepare source code of the Trigger Function:  
Create a zip archive containing triggerfunction/pom.xml, triggerfunction/src.

3. Create an App Engine application in the us-central region, it's a requirement to create a Cloud Scheduler job.

4. Create a service account with the project owner role (if doesn't exist). Generate a file with a key. 

5. Make sure the current gcloud configuration is configured to work with the project.

6. This step is only required for initial deployment, and not for updates. From the 
deployment/terraform directory run:  
terraform init

7. From the deployment/terraform directory run:  
   terraform apply -var project=<PROJECT_ID> \
   -var service-account-key-location="<PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE>" \
   -var env=<ENVIRONMENT_SUFFIX> \
   -var dataflow-system-files-bucket=<DATAFLOW_SYSTEM_FILES_BUCKET> \
   -var dataflow-temp-files-bucket=<DATAFLOW_TEMP_FILES_BUCKET> \
   -var function-system-files-bucket=<FUNCTION_SYSTEM_FILES_BUCKET> \
   -var dataset=<DATASET_NAME> \
   -var trigger-function-location=<PATH_TO_FUNCTION_SOURCE_ZIP_FILE> \
   -var output-bucket=<OUTPUT_BUCKET>

8. In BigQuery execute the following query:  
INSERT INTO `<DATASET_NAME>.chicago_boundaries` (boundaries) VALUES ((SELECT ST_GEOGFROMTEXT(the_geom) FROM <DATASET_NAME>.chicago_boundaries_raw LIMIT 1))

9. Deploy the Dataflow pipeline. 
   - Set default google account  
     export GOOGLE_APPLICATION_CREDENTIALS=<PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE>
   - From the bqtostoragedataflow directory run  
     mvn -Pdataflow-runner clean compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage \
     -Dexec.cleanupDaemonThreads=false -Dexec.args="\
     --project=<PROJECT_ID> \
     --tempLocation=gs://<TEMP_FILES_BUCKET>/public-bq-to-bq \
     --stagingLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-bq/staging \
     --runner=DataflowRunner \
     --region=us-central1 \
     --subnetwork=https://www.googleapis.com/compute/v1/projects/<PROJECT_ID>/regions/us-central1/subnetworks/dataflow \
     --templateLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-bq/template \
     --dataset=<DATASET_NAME> \
     --limit=2000000000 \
     --fromDate=2018-08-01 \
     --outputDirectory=gs://<OUTPUT_BUCKET>/trips \
     --flexRSGoal=COST_OPTIMIZED"

10. Disable the service account with the owner role. (if required)

