# Run the pipeline locally:
mvn clean compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage \
-Dexec.cleanupDaemonThreads=false -Dexec.args="\
    --project=<PROJECT_ID> \
    --tempLocation=gs://<TEMP_FILES_BUCKET>/public-bq-to-csv \
    --stagingLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-csv/staging \
    --runner=DirectRunner \
    --region=us-central1 \
    --subnetwork=https://www.googleapis.com/compute/v1/projects/<PROJECT_ID>/regions/us-central1/subnetworks/dataflow \
    --templateLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-csv/template \
    --dataset=<DATASET_NAME> \
    --outputDirectory=gs://<OUTPUT_BUCKET>/trips"

# Stage the pipeline in the Cloud Dataflow:
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
