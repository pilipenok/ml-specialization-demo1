# Files created with
gsutil iam get gs://<BUCKET_NAME> > <FILE_NAME>

# Steps to update a bucket IAM policy from command line
1. gsutil iam get gs://<BUCKET_NAME> > <FILE_NAME>
2. Modify the file <FILE_NAME>
3. gsutil iam set <FILE_NAME>  gs://<BUCKET_NAME>

