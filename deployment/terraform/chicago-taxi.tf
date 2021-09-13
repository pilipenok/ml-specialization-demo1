/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 *
 * This Terraform configuration includes only Data Pipeline set-up, and not ML part.
 */

variable "project" {
  type = string
}

variable "service-account-key-location" {
  type = string
}

variable "env" {
  type = string
}

variable "dataflow-system-files-bucket" {
  type = string
}

variable "function-system-files-bucket" {
  type = string
}

variable "dataflow-temp-files-bucket" {
  type = string
}

variable "output-bucket" {
  type = string
}

variable "dataset" {
  type = string
}

variable "trigger-function-location" {
  type = string
}

locals {
  multi-region        = "US"
  region              = "us-central1"
  zone                = "us-central1-c"
  app-engine-location = "us-central"
}

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.79.0"
    }
  }
}

provider "google" {
  credentials = file(var.service-account-key-location)
  project     = var.project
  region      = local.region
  zone        = local.zone
}

# Cloud Storage Buckets

resource "google_storage_bucket" "ChicagoTaxi" {
  name                        = var.output-bucket
  location                    = local.region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "REGIONAL"
}

resource "google_storage_bucket" "DataflowSystemFiles" {
  name                        = var.dataflow-system-files-bucket
  location                    = local.region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "REGIONAL"
}

resource "google_storage_bucket" "DataflowTempFiles" {
  name                        = var.dataflow-temp-files-bucket
  location                    = local.region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "REGIONAL"
}

resource "google_storage_bucket" "FunctionsSystemFiles" {
  name                        = var.function-system-files-bucket
  location                    = local.region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "REGIONAL"
}

# Cloud Storage Objects

resource "google_storage_bucket_object" "ChicagoBoundaries" {
  name   = "bq/City_Boundary.csv"
  source = "City_Boundary.csv"
  bucket = google_storage_bucket.DataflowSystemFiles.id
}

resource "google_storage_bucket_object" "NationalHolidays" {
  name   = "bq/National_Holidays.csv"
  source = "National_Holidays.csv"
  bucket = google_storage_bucket.DataflowSystemFiles.id
}

resource "google_storage_bucket_object" "TriggerFunctionSource" {
  name   = "triggerfunction.zip"
  source = var.trigger-function-location
  bucket = google_storage_bucket.FunctionsSystemFiles.id
}

# VPC

resource "google_compute_network" "Dataflow" {
  name                    = "dataflow"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "Dataflow" {
  project       = var.project
  name          = "dataflow"
  ip_cidr_range = "10.0.0.0/29"
  network       = google_compute_network.Dataflow.id
  region        = local.region
}

# BigQuery datasets

resource "google_bigquery_dataset" "ChicagoTaxi" {
  dataset_id                 = var.dataset
  location                   = local.multi-region
  delete_contents_on_destroy = false
}

# BigQuery tables

resource "google_bigquery_table" "TaxiTripsView" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "taxi_trips_view"
  deletion_protection = false
  view {
    use_legacy_sql = false
    query          = "SELECT * FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`"
  }
}

resource "google_bigquery_table" "ProcessedTrips" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "processed_trips"
  deletion_protection = true
  schema              = <<EOF
[
  {
    "mode": "REQUIRED",
    "name": "unique_key",
    "type": "STRING"
  },
  {
    "mode": "REQUIRED",
    "name": "processed_timestamp",
    "type": "TIMESTAMP"
  }
]
EOF
}

resource "google_bigquery_table" "ChicagoBoundariesRaw" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "chicago_boundaries_raw"
  deletion_protection = false
  depends_on = [
    google_storage_bucket_object.ChicagoBoundaries
  ]
  external_data_configuration {
    source_format = "CSV"
    autodetect    = true
    source_uris = [
      "gs://${google_storage_bucket_object.ChicagoBoundaries.bucket}/${google_storage_bucket_object.ChicagoBoundaries.output_name}"
    ]
  }
}

resource "google_bigquery_table" "ChicagoBoundaries" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "chicago_boundaries"
  deletion_protection = false
  schema              = <<EOF
[
  {
    "mode": "REQUIRED",
    "name": "boundaries",
    "type": "GEOGRAPHY"
  }
]
EOF
}

resource "google_bigquery_table" "NationalHolidays" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "national_holidays"
  deletion_protection = false
  depends_on = [
    google_storage_bucket_object.NationalHolidays
  ]
  external_data_configuration {
    source_format = "CSV"
    autodetect    = true
    source_uris = [
      "gs://${google_storage_bucket_object.NationalHolidays.bucket}/${google_storage_bucket_object.NationalHolidays.output_name}"
    ]
  }
}

resource "google_bigquery_table" "PublicDatasetTables" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "public_dataset_tables"
  deletion_protection = false
  view {
    use_legacy_sql = false
    query          = "SELECT * FROM bigquery-public-data.chicago_taxi_trips.__TABLES__"
  }
}

# IAM Roles

resource "google_project_iam_custom_role" "ChicagoTaxiDataflow" {
  role_id     = "ChicagoTaxiDataflow"
  title       = "Chicago Taxi Dataflow"
  description = "A role for a Dataflow controller service account."
  permissions = [
    "bigquery.datasets.create",
    "bigquery.datasets.get",
    "bigquery.jobs.create",
    "bigquery.readsessions.create",
    "bigquery.readsessions.getData",
    "bigquery.readsessions.update"
  ]
}

resource "google_project_iam_custom_role" "ChicagoTaxiBQWriter" {
  role_id     = "ChicagoTaxiBQWriter"
  title       = "Chicago Taxi Dataflow BQ ProcessedTrips Policy"
  description = "Provides permissions to Dataflow pipeline to do inserts into a BigQuery table"
  permissions = [
    "bigquery.tables.get",
    "bigquery.tables.getData",
    "bigquery.tables.updateData"
  ]
}

resource "google_project_iam_custom_role" "ChicagoTaxiBQReader" {
  role_id     = "ChicagoTaxiBQReader"
  title       = "Chicago Taxi BQ Read-Only Policy"
  description = "Chicago Taxi BQ Read-Only Policy"
  permissions = [
    "bigquery.tables.get",
    "bigquery.tables.getData"
  ]
}

resource "google_project_iam_custom_role" "ChicagoTaxiTriggerFunction" {
  role_id     = "ChicagoTaxiTriggerFunction"
  title       = "Chicago Taxi Dataflow Runner"
  description = "Chicago Taxi Dataflow Runner"
  permissions = [
    "compute.machineTypes.get",
    "dataflow.jobs.create",
    "iam.serviceAccounts.actAs",
    "resourcemanager.projects.get",
    "storage.objects.get",
    "storage.objects.list"
  ]
}

resource "google_project_iam_custom_role" "ChicagoTaxiStorageWriter" {
  role_id     = "ChicagoTaxiStorageWriter"
  title       = "Chicago Taxi Dataflow Storage CSV Creator"
  description = "Allows a Dataflow pipeline to do inserts into a BigQuery table"
  permissions = [
    "storage.buckets.get",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.getIamPolicy",
    "storage.objects.list"
  ]
}

resource "google_project_iam_custom_role" "ChicagoTaxiDataflowStorageSystemAccess" {
  role_id     = "ChicagoTaxiDataflowStorageSystemAccess"
  title       = "Chicago Taxi Dataflow Storage System Access"
  description = "Allows a Dataflow to create/read the pipeline template and create/delete temporary files."
  permissions = [
    "storage.buckets.get",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.getIamPolicy",
    "storage.objects.list"
  ]
}

# Service Accounts

resource "google_service_account" "Dataflow" {
  account_id   = "chicago-taxi-dataflow"
  display_name = "chicago-taxi-dataflow"
  description  = "A service account runs Dataflow pipeline"
}

resource "google_service_account" "TriggerFunction" {
  account_id   = "chicago-taxi-trigger-functio"
  display_name = "chicago-taxi-trigger-function"
  description  = "A service account for Cloud Function TriggerFunction"
}

resource "google_service_account" "PipelineScheduler" {
  account_id   = "chicago-taxi-pipeline-sched"
  display_name = "chicago-taxi-pipeline-scheduler"
  description  = "A service account to run Cloud Scheduler"
}

# Service Accounts IAM binding

## Permissions binding for service account that runs Dataflow.

resource "google_project_iam_member" "DataflowWorker" {
  role   = "roles/dataflow.worker"
  member = "serviceAccount:${google_service_account.Dataflow.email}"
}

resource "google_project_iam_member" "Dataflow" {
  role   = google_project_iam_custom_role.ChicagoTaxiDataflow.id
  member = "serviceAccount:${google_service_account.Dataflow.email}"
}

### BigQuery IAM

resource "google_bigquery_table_iam_member" "TaxiTripsView_DataflowReadOnly" {
  dataset_id = google_bigquery_table.TaxiTripsView.dataset_id
  table_id   = google_bigquery_table.TaxiTripsView.id
  role       = google_project_iam_custom_role.ChicagoTaxiBQReader.id
  member     = "serviceAccount:${google_service_account.Dataflow.email}"
}

resource "google_bigquery_table_iam_member" "ChicagoBoundariesTable_DataflowReadOnly" {
  dataset_id = google_bigquery_table.ChicagoBoundaries.dataset_id
  table_id   = google_bigquery_table.ChicagoBoundaries.id
  role       = google_project_iam_custom_role.ChicagoTaxiBQReader.id
  member     = "serviceAccount:${google_service_account.Dataflow.email}"
}

resource "google_bigquery_table_iam_member" "NationalHolidaysTable_DataflowReadOnly" {
  dataset_id = google_bigquery_table.NationalHolidays.dataset_id
  table_id   = google_bigquery_table.NationalHolidays.id
  role       = google_project_iam_custom_role.ChicagoTaxiBQReader.id
  member     = "serviceAccount:${google_service_account.Dataflow.email}"
}

resource "google_bigquery_table_iam_member" "DataflowAccessToProcessedTripsTable" {
  dataset_id = google_bigquery_table.ProcessedTrips.dataset_id
  table_id   = google_bigquery_table.ProcessedTrips.id
  role       = google_project_iam_custom_role.ChicagoTaxiBQWriter.id
  member     = "serviceAccount:${google_service_account.Dataflow.email}"
}

### Cloud Storage IAM

resource "google_storage_bucket_iam_member" "DataflowAccessToSystemStorageFiles" {
  bucket = google_storage_bucket.DataflowSystemFiles.id
  role   = google_project_iam_custom_role.ChicagoTaxiDataflowStorageSystemAccess.id
  member = "serviceAccount:${google_service_account.Dataflow.email}"
}

resource "google_storage_bucket_iam_member" "DataflowAccessToTempStorage" {
  bucket = google_storage_bucket.DataflowTempFiles.id
  role   = google_project_iam_custom_role.ChicagoTaxiDataflowStorageSystemAccess.id
  member = "serviceAccount:${google_service_account.Dataflow.email}"
}

resource "google_storage_bucket_iam_member" "DataflowAccessToOutputStorage" {
  bucket = google_storage_bucket.ChicagoTaxi.id
  role   = google_project_iam_custom_role.ChicagoTaxiStorageWriter.id
  member = "serviceAccount:${google_service_account.Dataflow.email}"
  condition {
    title      = "CSV file folder"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.ChicagoTaxi.id}/objects/trips/\")"
  }
}

## Permissions binding for service account that runs Cloud Scheduler.

resource "google_cloudfunctions_function_iam_member" "SchedulerAccessToInvokeTriggerFunction" {
  cloud_function = google_cloudfunctions_function.TriggerFunction.name
  role           = "roles/cloudfunctions.invoker"
  member         = "serviceAccount:${google_service_account.PipelineScheduler.email}"
}

## Permissions binding for service account that runs Cloud Function.

resource "google_project_iam_member" "TriggerFunctionToCreateDataflowJob" {
  role   = google_project_iam_custom_role.ChicagoTaxiTriggerFunction.id
  member = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_project_iam_member" "TriggerFunctionToReadBigQuery" {
  role   = "roles/bigquery.dataViewer"
  member = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_project_iam_member" "TriggerFunctionToRunBigQueryJobs" {
  role   = "roles/bigquery.jobUser"
  member = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

### BigQuery IAM

resource "google_bigquery_table_iam_member" "TaxiId_TriggerFunctionReadOnly" {
  dataset_id = google_bigquery_table.ProcessedTrips.dataset_id
  table_id   = google_bigquery_table.ProcessedTrips.id
  role       = google_project_iam_custom_role.ChicagoTaxiBQReader.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

# Trigger Function

resource "google_cloudfunctions_function" "TriggerFunction" {
  name                  = "chicago-taxi-trigger-function"
  runtime               = "java11"
  available_memory_mb   = 256
  trigger_http          = true
  entry_point           = "com.epam.gcp.chicagotaximl.triggerfunction.DataflowTriggerHttpFunction"
  ingress_settings      = "ALLOW_ALL"
  service_account_email = google_service_account.TriggerFunction.email
  max_instances         = 1
  source_archive_bucket = google_storage_bucket_object.TriggerFunctionSource.bucket
  source_archive_object = google_storage_bucket_object.TriggerFunctionSource.name
}

# Cloud Scheduler to run TriggerFunction

resource "google_cloud_scheduler_job" "TriggerFunction" {
  name             = "chicago-taxi-trigger"
  schedule         = "0 21 * * *"
  time_zone        = "Atlantic/St_Helena" // UTC
  attempt_deadline = "320s"
  retry_config {
    retry_count = 0
  }
  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions_function.TriggerFunction.https_trigger_url
    headers = {
      "Content-Type" : "application/json"
    }
    oidc_token {
      service_account_email = google_service_account.PipelineScheduler.email
    }
    body = base64encode(
      <<EOF
      {
        "project": "${var.project}",
        "dataflow-job-name": "chicago-taxi-public-bq-to-csv",
        "gcs-path": "gs://${google_storage_bucket.DataflowSystemFiles.id}/public-bq-to-bq/template",
        "temp-location": "gs://${google_storage_bucket.DataflowTempFiles.id}/public-bq-to-bq",
        "service-account": "${google_service_account.Dataflow.email}",
        "region": "${local.region}",
        "check-last-modified-time": true,
        "dataset": "${google_bigquery_dataset.ChicagoTaxi.dataset_id}"
    }
    EOF
    )
  }
}
