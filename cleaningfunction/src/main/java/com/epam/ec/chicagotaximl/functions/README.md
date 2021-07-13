# To run locally (Cloud Storage permissions required):
mvn function:run

# Deployment
gcloud functions deploy chicago-taxi-ml-demo-1-function-1 \
    --entry-point=com.epam.ec.chicagotaximl.functions.CleaningHttp \
    --runtime=java11 \
    --trigger-http \
    --memory=128MB \
    --security-level=secure-optional \
    --service-account=chicago-taxi-ml-demo-1-functio@o-epm-gcp-by-meetup1-ml-t1iylu.iam.gserviceaccount.com
    --max-instances=1
