# Run the pipeline locally:
mvn clean compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage \
-Dexec.cleanupDaemonThreads=false -Dexec.args="\
    --project=o-epm-gcp-by-meetup1-ml-t1iylu \
    --tempLocation=gs://chicago-taxi-dataflow-temp-files/public-bq-to-bq \
    --stagingLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/staging \
    --runner=DirectRunner \
    --region=us-central1 \
    --subnetwork=https://www.googleapis.com/compute/v1/projects/o-epm-gcp-by-meetup1-ml-t1iylu/regions/us-central1/subnetworks/dataflow \
    --templateLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/template \
    --sourceTable=chicago_taxi_ml_demo_1.taxi_trips_sample \
    --limit=10000 \
    --fromDate=2018-08-01 \
    --destinationTable=chicago_taxi_ml_demo_1.taxi_id \
    --outputDirectory=gs://chicago-taxi-ml-demo-1/trips"

# Stage the pipeline in the Cloud Dataflow:
mvn -Pdataflow-runner clean compile exec:java -Dexec.mainClass=com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage \
-Dexec.cleanupDaemonThreads=false -Dexec.args="\
    --project=o-epm-gcp-by-meetup1-ml-t1iylu \
    --tempLocation=gs://chicago-taxi-dataflow-temp-files/public-bq-to-bq \
    --stagingLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/staging \
    --runner=DataflowRunner \
    --region=us-central1 \
    --subnetwork=https://www.googleapis.com/compute/v1/projects/o-epm-gcp-by-meetup1-ml-t1iylu/regions/us-central1/subnetworks/dataflow \
    --templateLocation=gs://chicago-taxi-dataflow/public-bq-to-bq/template \
    --sourceTable=chicago_taxi_ml_demo_1.taxi_trips_view \
    --limit=2000000000 \
    --fromDate=2018-08-01 \
    --destinationTable=chicago_taxi_ml_demo_1.taxi_id \
    --outputDirectory=gs://chicago-taxi-ml-demo-1/trips \
    --flexRSGoal=COST_OPTIMIZED"

# Permissions
Dataflow runs the pipeline with the service account (Dataflow Service Account / Controller Account) 
chicago-taxi-ml-demo-1-dataflo@PROJECT_ID.iam.gserviceaccount.com. 
The service account is bonded to the following roles: 
- dataflow.worker and ChicagoTaxiDataflow are assigned directly to the account,
- ChicagoTaxiDataflowBqReadOnlyPolicy is bonded through a BigQuery Table-level access control for the view 'taxi_trips_view' 
  and tables 'chicago_boundaries' and 'national_holidays',
- ChicagoTaxiDataflowBQTaxiIdPolicy is bonded through a BigQuery Table-level access control for the table 'taxi_id',
- ChicagoTaxiDataflowStorageCsvCreator is bonded through a Storage IAM for the bucket/folder chicago-taxi-ml-demo-1/trips/
- ChicagoTaxiDataflowStorageSystemAccess is bonded through a Storage IAM for the buckets chicago-taxi-temp-files and chicago-taxi-dataflow
- storage.objectViewer is bonded through a Storage IAM for bucket chicago-taxi-dataflow
