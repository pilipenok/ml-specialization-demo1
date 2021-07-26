# Run the pipeline locally:
mvn compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage -Dexec.cleanupDaemonThreads=false -Dexec.args="\
    --project=o-epm-gcp-by-meetup1-ml-t1iylu \
    --tempLocation=gs://chicago-taxi-dataflow-temp-files/public-bq-to-bq \
    --stagingLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/staging \
    --runner=DirectRunner \
    --region=us-central1 \
    --templateLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/template \
    --sourceTable=chicago_taxi_ml_demo_1.taxi_trips_sample \
    --limit=10000 \
    --interval=5000 \
    --destinationTable=chicago_taxi_ml_demo_1.taxi_id \
    --outputDirectory=gs://chicago-taxi-ml-demo-1/trips"

# Stage the pipeline in the Cloud Dataflow:
mvn -Pdataflow-runner exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage -Dexec.cleanupDaemonThreads=false -Dexec.args="\
    --project=o-epm-gcp-by-meetup1-ml-t1iylu \
    --tempLocation=gs://chicago-taxi-dataflow-temp-files/public-bq-to-bq \
    --stagingLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/staging \
    --runner=DataflowRunner \
    --region=us-central1 \
    --templateLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/template \
    --sourceTable=chicago_taxi_ml_demo_1.taxi_trips_view \
    --limit=2000000000 \
    --interval=1100 \
    --destinationTable=chicago_taxi_ml_demo_1.taxi_id \
    --outputDirectory=gs://chicago-taxi-ml-demo-1/trips"

# Permissions
Dataflow runs the pipeline with the service account (controller account) chicago-taxi-ml-demo-1-dataflo@PROJECT_ID.iam.gserviceaccount.com. 
The service account is assigned the following roles: 
- dataflow.worker and ChicagoTaxiDataflow are assigned directly to an account,
- ChicagoTaxiDataflowBqReadOnlyPolicy is assigned through a BigQuery Table-level access control for view 'taxi_trips_view' 
  and tables 'chicago_boundaries' and 'national_holidays',
- ChicagoTaxiDataflowBQTaxiIdPolicy is assigned through a BigQuery Table-level access control for table 'taxi_id',
- ChicagoTaxiDataflowStorageCsvCreator is assigned through a Storage IAM for a bucket/folder chicago-taxi-ml-demo-1/trips/
- ChicagoTaxiDataflowStorageSystemAccess is assigned through a Storage IAM for a bucket chicago-taxi-temp-files
- storage.objectViewer is assigned through a Storage IAM for a bucket chicago-taxi-dataflow
