# Run the pipeline locally:
mvn clean compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage \
-Dexec.cleanupDaemonThreads=false -Dexec.args="\
    --project=<PROJECT_ID> \
    --tempLocation=gs://<TEMP_FILES_BUCKET>/public-bq-to-bq \
    --stagingLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-bq/staging \
    --runner=DirectRunner \
    --region=us-central1 \
    --subnetwork=https://www.googleapis.com/compute/v1/projects/<PROJECT_ID>/regions/us-central1/subnetworks/dataflow \
    --templateLocation=gs://<SYSTEM_FILES_BUCKET>/public-bq-to-bq/template \
    --dataset=<DATASET_NAME> \
    --limit=10000 \
    --fromDate=2018-08-01 \
    --outputDirectory=gs://<OUTPUT_BUCKET>/trips"

# Stage the pipeline in the Cloud Dataflow:
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
