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

variable "cleanup-function-location" {
  type = string
}

variable "source-table" {
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

// Storage for Dataflow template and deployment files
resource "google_storage_bucket" "DataflowSystemFiles" {
  name                        = var.dataflow-system-files-bucket
  location                    = local.region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "REGIONAL"
}

// Storage for Dataflow temporary files
resource "google_storage_bucket" "DataflowTempFiles" {
  name                        = var.dataflow-temp-files-bucket
  location                    = local.region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "REGIONAL"
}

// Storage for Cloud Functions deployment source files
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

// Source code of TriggerFunction
resource "google_storage_bucket_object" "TriggerFunctionSource" {
  name   = "triggerfunction.zip"
  source = var.trigger-function-location
  bucket = google_storage_bucket.FunctionsSystemFiles.id
}

// Source code of CleanupFunction
resource "google_storage_bucket_object" "CleanupFunctionSource" {
  name   = "cleanupfunction.zip"
  source = var.cleanup-function-location
  bucket = google_storage_bucket.FunctionsSystemFiles.id
}

// Topic to execute TriggerFunction from Cloud Scheduler
resource "google_pubsub_topic" "PipelineTrigger" {
  name = "chicago-taxi-trigger"
}

// Network for Dataflow job
resource "google_compute_network" "TaxiDataflow" {
  name                    = "chicago-taxi-dataflow"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "TaxiDataflow" {
  project       = var.project
  name          = "dataflow"
  ip_cidr_range = "10.0.0.0/29"
  network       = google_compute_network.TaxiDataflow.id
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
    "mode": "NULLABLE",
    "name": "processed_timestamp",
    "type": "TIMESTAMP"
  }
]
EOF
}

resource "google_bigquery_table" "SourceModificationTimestamps" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "source_modification_timestamps"
  description         = "last_modified_time of table 'taxi_trips' from `${google_bigquery_table.PublicDatasetTables.dataset_id}.${google_bigquery_table.PublicDatasetTables.table_id}`"
  deletion_protection = true
  schema              = <<EOF
[
  {
    "mode": "REQUIRED",
    "name": "modification_timestamp",
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
  view {
    use_legacy_sql = false
    query          = "SELECT ST_GEOGFROMTEXT(the_geom) AS boundaries FROM `${google_bigquery_table.ChicagoBoundariesRaw.dataset_id}.${google_bigquery_table.ChicagoBoundariesRaw.table_id}`"
  }
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

resource "google_bigquery_table" "MlDataset" {
  dataset_id          = google_bigquery_dataset.ChicagoTaxi.dataset_id
  table_id            = "ml_dataset"
  deletion_protection = false
  // The table is going to be recreated with proper fields by Trigger Function
  schema              = <<EOF
[
  {
    "mode": "NULLABLE",
    "name": "area",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "year",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "quarter",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "quarter_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "quarter_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "quarter_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "month",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "month_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "month_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "month_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "day_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "hour",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "hour_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "hour_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "hour_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day_period",
    "type": "STRING"
  },
  {
    "mode": "NULLABLE",
    "name": "week",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "week_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "week_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "week_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day_of_week",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "day_of_week_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day_of_week_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "day_of_week_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "weekday_hour_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "weekday_hour_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "weekday_hour_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "yearday_hour_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "yearday_hour_cos",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "yearday_hour_sin",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "is_weekend",
    "type": "BOOLEAN"
  },
  {
    "mode": "NULLABLE",
    "name": "is_holiday",
    "type": "BOOLEAN"
  },
  {
    "mode": "NULLABLE",
    "name": "n_trips",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "n_trips_num",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "log_n_trips",
    "type": "FLOAT"
  },
  {
    "mode": "NULLABLE",
    "name": "trips_bucket",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "trips_bucket_num",
    "type": "FLOAT"
  }
]
EOF
}

# Common IAM Roles

resource "google_project_iam_custom_role" "BigQueryWriter" {
  role_id     = "ChicagoTaxiBQWriter"
  title       = "Chicago Taxi BigQuery Writer"
  description = "Provides permissions to write into a BigQuery tables"
  permissions = [
    "bigquery.tables.get",
    "bigquery.tables.getData",
    "bigquery.tables.updateData"
  ]
}

resource "google_project_iam_custom_role" "BigQueryReader" {
  role_id     = "ChicagoTaxiBQReader"
  title       = "Chicago Taxi BigQuery Reader"
  description = "Provides permissions to query a BigQuery tables"
  permissions = [
    "bigquery.tables.get",
    "bigquery.tables.getData"
  ]
}

resource "google_project_iam_custom_role" "StorageWriter" {
  role_id     = "ChicagoTaxiStorageWriter"
  title       = "Chicago Taxi Storage Writer"
  description = "Provides permissions to write to Cloud Storage"
  permissions = [
    "storage.buckets.get",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.getIamPolicy",
    "storage.objects.list"
  ]
}

# PipelineScheduler Service Account and IAM

resource "google_service_account" "PipelineScheduler" {
  account_id   = "chicago-taxi-pipeline-sched"
  display_name = "chicago-taxi-pipeline-scheduler"
  description  = "A service account runs Scheduler ${google_cloud_scheduler_job.TriggerFunction.name}"
}

resource "google_pubsub_topic_iam_binding" "PipelineScheduler_PubSubPublisher" {
  topic = google_pubsub_topic.PipelineTrigger.name
  role  = "roles/pubsub.publisher"
  members = [
    "serviceAccount:${google_service_account.PipelineScheduler.email}"
  ]
}

# TriggerFunction Service Account and IAM

resource "google_service_account" "TriggerFunction" {
  account_id   = "chicago-taxi-trigger-functio"
  display_name = "chicago-taxi-trigger-function"
  description  = "A service account runs TriggerFunction"
}

resource "google_project_iam_custom_role" "TriggerFunction" {
  role_id     = "ChicagoTaxiTriggerFunction"
  title       = "Chicago Taxi Trigger Function"
  description = "A role for a service account that runs TriggerFunction."
  permissions = [
    "dataflow.jobs.create",
    "iam.serviceAccounts.actAs",
    "resourcemanager.projects.get",
    "bigquery.jobs.create"
  ]
}

resource "google_project_iam_member" "TriggerFunction" {
  role   = google_project_iam_custom_role.TriggerFunction.id
  member = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_TaxiTripsViewReader" {
  dataset_id = google_bigquery_table.TaxiTripsView.dataset_id
  table_id   = google_bigquery_table.TaxiTripsView.id
  role       = google_project_iam_custom_role.BigQueryReader.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_PublicDatasetTablesReader" {
  dataset_id = google_bigquery_table.PublicDatasetTables.dataset_id
  table_id   = google_bigquery_table.PublicDatasetTables.id
  role       = google_project_iam_custom_role.BigQueryReader.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_ProcessedTripsWriter" {
  dataset_id = google_bigquery_table.ProcessedTrips.dataset_id
  table_id   = google_bigquery_table.ProcessedTrips.id
  role       = google_project_iam_custom_role.BigQueryWriter.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_SourceTimestampWriter" {
  dataset_id = google_bigquery_table.SourceModificationTimestamps.dataset_id
  table_id   = google_bigquery_table.SourceModificationTimestamps.id
  role       = google_project_iam_custom_role.BigQueryWriter.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_ChicagoBoundariesReader" {
  dataset_id = google_bigquery_table.ChicagoBoundaries.dataset_id
  table_id   = google_bigquery_table.ChicagoBoundaries.id
  role       = google_project_iam_custom_role.BigQueryReader.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_ChicagoBoundariesRawReader" {
  dataset_id = google_bigquery_table.ChicagoBoundariesRaw.dataset_id
  table_id   = google_bigquery_table.ChicagoBoundariesRaw.id
  role       = google_project_iam_custom_role.BigQueryReader.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_NationalHolidaysReader" {
  dataset_id = google_bigquery_table.NationalHolidays.dataset_id
  table_id   = google_bigquery_table.NationalHolidays.id
  role       = google_project_iam_custom_role.BigQueryReader.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_bigquery_table_iam_member" "TriggerFunction_MlDatasetWriter" {
  dataset_id = google_bigquery_table.MlDataset.dataset_id
  table_id   = google_bigquery_table.MlDataset.id
  role       = google_project_iam_custom_role.BigQueryWriter.id
  member     = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

resource "google_storage_bucket_iam_member" "TriggerFunction_SystemStorageReader" {
  bucket = google_storage_bucket.DataflowSystemFiles.id
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.TriggerFunction.email}"
}

# Dataflow Service Account and IAM

resource "google_service_account" "TripsDataflow" {
  account_id   = "chicago-taxi-dataflow"
  display_name = "chicago-taxi-dataflow"
  description  = "A service account runs TripsDataflow"
}

resource "google_project_iam_custom_role" "TripsDataflow" {
  role_id     = "ChicagoTaxiDataflow"
  title       = "Chicago Taxi Dataflow"
  description = "A role for a Dataflow worker service account."
  permissions = [
    "bigquery.datasets.create",
    "bigquery.jobs.create",
    "bigquery.readsessions.create",
    "bigquery.readsessions.getData",
  ]
}

resource "google_project_iam_member" "TripsDataflow_Worker" {
  role   = "roles/dataflow.worker"
  member = "serviceAccount:${google_service_account.TripsDataflow.email}"
}

resource "google_project_iam_member" "TripsDataflow" {
  role   = google_project_iam_custom_role.TripsDataflow.id
  member = "serviceAccount:${google_service_account.TripsDataflow.email}"
}

resource "google_bigquery_table_iam_member" "TripsDataflow_MlDatasetReader" {
  dataset_id = google_bigquery_table.MlDataset.dataset_id
  table_id   = google_bigquery_table.MlDataset.id
  role       = google_project_iam_custom_role.BigQueryReader.id
  member     = "serviceAccount:${google_service_account.TripsDataflow.email}"
}

resource "google_storage_bucket_iam_member" "TripsDataflow_TempStorageWriter" {
  bucket = google_storage_bucket.DataflowTempFiles.id
  role   = google_project_iam_custom_role.StorageWriter.id
  member = "serviceAccount:${google_service_account.TripsDataflow.email}"
}

resource "google_storage_bucket_iam_member" "TripsDataflow_OutputStorageWriter" {
  bucket = google_storage_bucket.ChicagoTaxi.id
  role   = google_project_iam_custom_role.StorageWriter.id
  member = "serviceAccount:${google_service_account.TripsDataflow.email}"
  condition {
    title      = "CSV file folder"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.ChicagoTaxi.id}/objects/trips/\")"
  }
}

# CleanupFunction Service Account and IAM

resource "google_service_account" "CleanupFunction" {
  account_id   = "chicago-taxi-cleanup-functio"
  display_name = "chicago-taxi-cleanup-function"
  description  = "A service account runs CleanupFunction"
}

resource "google_project_iam_custom_role" "CleanupFunction" {
  role_id     = "ChicagoTaxiCleanupFunction"
  title       = "Chicago Taxi Cleanup Function"
  description = "A role for a service account that runs CleanupFunction"
  permissions = [
    "bigquery.jobs.create"
  ]
}

resource "google_project_iam_member" "CleanupFunction" {
  role   = google_project_iam_custom_role.CleanupFunction.id
  member = "serviceAccount:${google_service_account.CleanupFunction.email}"
}

resource "google_bigquery_table_iam_member" "CleanupFunction_ProcessedTripsWriter" {
  dataset_id = google_bigquery_table.ProcessedTrips.dataset_id
  table_id   = google_bigquery_table.ProcessedTrips.id
  role       = google_project_iam_custom_role.BigQueryWriter.id
  member     = "serviceAccount:${google_service_account.CleanupFunction.email}"
}

resource "google_bigquery_table_iam_member" "CleanupFunction_MlDatasetWriter" {
  dataset_id = google_bigquery_table.MlDataset.dataset_id
  table_id   = google_bigquery_table.MlDataset.id
  role       = google_project_iam_custom_role.BigQueryWriter.id
  member     = "serviceAccount:${google_service_account.CleanupFunction.email}"
}

# Trigger Function
resource "google_cloudfunctions_function" "TriggerFunction" {
  name                  = "chicago-taxi-trigger-function"
  runtime               = "java11"
  available_memory_mb   = 256
  entry_point           = "com.epam.gcp.chicagotaximl.triggerfunction.TriggerFunctionPubSubEvent"
  service_account_email = google_service_account.TriggerFunction.email
  max_instances         = 1
  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.PipelineTrigger.id
  }
  depends_on = [
    google_storage_bucket_object.TriggerFunctionSource
  ]
  source_archive_bucket = google_storage_bucket_object.TriggerFunctionSource.bucket
  source_archive_object = google_storage_bucket_object.TriggerFunctionSource.name
  environment_variables = {
    project           = var.project
    dataflow_job_name = "chicago-taxi-public-bq-to-csv"
    gcs_path          = "gs://${google_storage_bucket.DataflowSystemFiles.id}/public-bq-to-csv/template"
    temp_location     = "gs://${google_storage_bucket.DataflowTempFiles.id}/public-bq-to-csv"
    service_account   = google_service_account.TripsDataflow.email
    region            = local.region
    dataset           = google_bigquery_dataset.ChicagoTaxi.dataset_id
    source_table      = var.source-table
    start_date        = "2020-04-01"
  }
}

# Cleanup Function
resource "google_cloudfunctions_function" "CleanupFunction" {
  name                  = "chicago-taxi-cleanup-function"
  runtime               = "java11"
  available_memory_mb   = 256
  entry_point           = "com.epam.gcp.chicagotaximl.cleanupfunction.CleanupFunctionStorageEvent"
  service_account_email = google_service_account.CleanupFunction.email
  max_instances         = 1
  event_trigger {
    event_type = "google.storage.object.finalize"
    resource   = google_storage_bucket.ChicagoTaxi.id
  }
  depends_on = [
    google_storage_bucket_object.CleanupFunctionSource
  ]
  source_archive_bucket = google_storage_bucket_object.CleanupFunctionSource.bucket
  source_archive_object = google_storage_bucket_object.CleanupFunctionSource.name
  environment_variables = {
    dataset  = google_bigquery_dataset.ChicagoTaxi.dataset_id
    filename = "trips/trips.csv"
  }
}

# Cloud Scheduler run TriggerFunction via Pub/Sub
resource "google_cloud_scheduler_job" "TriggerFunction" {
  name        = "chicago-taxi-trigger"
  schedule    = "0 4 * * *"
  description = "Runs TriggerFunction via Pub/Sub topic ${google_pubsub_topic.PipelineTrigger.name}"
  time_zone   = "Atlantic/St_Helena" // UTC
  pubsub_target {
    topic_name = google_pubsub_topic.PipelineTrigger.id
    data       = base64encode("1")
  }
}
