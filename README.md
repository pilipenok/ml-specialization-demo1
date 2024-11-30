# Cloud environment setup
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
Create a zip archive triggerfunction.zip containing triggerfunction/pom.xml, triggerfunction/src. 

3. Prepare source code of the Cleanup Function:  
Create a zip archive cleanupfunction.zip containing cleanupfunction/pom.xml, cleanupfunction/src. 

4. Create an App Engine application in the us-central region, it's a requirement to create a Cloud Scheduler job.

5. Create a service account with the project owner role (if doesn't exist). Generate a file with a key. 

6. Make sure the current gcloud configuration is configured to work with the project.

7. Initialize terraform. From the deployment/terraform directory run:  
terraform init

8. From the deployment directory run:  
   terraform apply -var project=<PROJECT_ID> \
   -var service-account-key-location="<PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE>" \
   -var dataflow-system-files-bucket=<DATAFLOW_SYSTEM_FILES_BUCKET> \
   -var dataflow-temp-files-bucket=<DATAFLOW_TEMP_FILES_BUCKET> \
   -var function-system-files-bucket=<FUNCTION_SYSTEM_FILES_BUCKET> \
   -var output-bucket=<OUTPUT_BUCKET> \
   -var dataset=<DATASET_NAME> \
   -var source-table=<SOURCE_TABLE_NAME> \
   -var trigger-function-location=<PATH_TO_TRIGGER_FUNCTION_SOURCE_ZIP_FILE> \
   -var cleanup-function-location=<PATH_TO_CLEANUP_FUNCTION_SOURCE_ZIP_FILE>

9. Deploy the Dataflow pipeline. 
   - Set default google account  
     export GOOGLE_APPLICATION_CREDENTIALS=<PATH_TO_SERVICE_ACCOUNT_KEY_WITH_OWNER_ROLE>
   - From the bqtostoragedataflow directory run  
     mvn -Pdataflow-runner clean compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage \
     -Dexec.cleanupDaemonThreads=false -Dexec.args="\
     --project=<PROJECT_ID> \
     --tempLocation=gs://<TEMP_FILES_BUCKET>/public-bq-to-csv \
     --stagingLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-csv/staging \
     --runner=DataflowRunner \
     --region=us-central1 \
     --subnetwork=https://www.googleapis.com/compute/v1/projects/<PROJECT_ID>/regions/us-central1/subnetworks/dataflow \
     --templateLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-csv/template \
     --dataset=<DATASET_NAME> \
     --outputDirectory=gs://<OUTPUT_BUCKET>/trips"

10. Disable the service account with the owner role. (if required)

