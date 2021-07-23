# Service account to run Dataflow pipeline.
# Replace PROJECT_ID with correct ID of the project.

gcloud iam service-accounts create chicago-taxi-ml-demo-1-dataflo \
    --description="Service account to run Dataflow pipeline" \
    --display-name="chicago-taxi-ml-demo-1-dataflow";

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:chicago-taxi-ml-demo-1-dataflo@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/dataflow.worker";

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:chicago-taxi-ml-demo-1-dataflo@PROJECT_ID.iam.gserviceaccount.com" \
    --role="projects/PROJECT_ID/roles/ChicagoTaxiDataflow";

