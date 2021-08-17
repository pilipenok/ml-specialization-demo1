# Run locally
mvn function:run

# Deploy from local machine. Run from the project root
gcloud functions deploy chicago-taxi-ml-demo-1-trigger \
--entry-point=com.epam.gcp.chicagotaximl.triggerfunction.DataflowTriggerHttpFunction  \
--runtime=java11 \
--trigger-http \
--memory=256MB \
--security-level=secure-always \
--service-account=chicago-taxi-ml-deml-1-trigger@o-epm-gcp-by-meetup1-ml-t1iylu.iam.gserviceaccount.com \
--max-instances=1 \
--ingress-settings=all
