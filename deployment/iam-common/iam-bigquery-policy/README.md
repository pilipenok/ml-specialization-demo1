# Files created with
bq get-iam-policy <PROJECT_ID>:<DATASET_NAME>.<TABLE_NAME> > <FILE_NAME>

# Steps to update a BigQuery table IAM policy from command line
1. bq get-iam-policy <PROJECT_ID>:<DATASET_NAME>.<TABLE_NAME> > <FILE_NAME>
2. Modify the file <FILE_NAME>
3. bq set-iam-policy <PROJECT_ID>:<DATASET_NAME>.<TABLE_NAME> <FILE_NAME>

