/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.cleanupfunction.integration;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.DatasetInfo;
import com.google.cloud.bigquery.Field;
import com.google.cloud.bigquery.FieldValueList;
import com.google.cloud.bigquery.FormatOptions;
import com.google.cloud.bigquery.Job;
import com.google.cloud.bigquery.JobStatistics.LoadStatistics;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.Schema;
import com.google.cloud.bigquery.StandardSQLTypeName;
import com.google.cloud.bigquery.StandardTableDefinition;
import com.google.cloud.bigquery.TableDataWriteChannel;
import com.google.cloud.bigquery.TableId;
import com.google.cloud.bigquery.TableInfo;
import com.google.cloud.bigquery.TableResult;
import com.google.cloud.bigquery.WriteChannelConfiguration;
import com.google.cloud.bigquery.testing.RemoteBigQueryHelper;
import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import lombok.Getter;

/**
 * Helper class for integration testing with BigQuery.
 */
public class BigQueryTesting implements Closeable {

  private static final String PROCESSED_TRIPS_TABLE = "processed_trips";
  private static final String ML_DATASET_TABLE = "ml_dataset";

  private final RemoteBigQueryHelper bigqueryHelper = RemoteBigQueryHelper.create();

  @Getter
  private final BigQuery bigquery = bigqueryHelper.getOptions().getService();

  @Getter
  private final String dataset = RemoteBigQueryHelper.generateDatasetName();

  private static final List<Field> PROCESSED_TRIPS_FIELDS = Arrays.asList(
      Field.of("unique_key", StandardSQLTypeName.STRING),
      Field.of("processed_timestamp", StandardSQLTypeName.TIMESTAMP));

  private static final List<Field> ML_DATASET_FIELDS = Arrays.asList(
      Field.of("some_field", StandardSQLTypeName.STRING));

  private TableId processedTripsTable;
  private TableId mlDatasetTable;

  public BigQueryTesting() {
    bigquery.create(DatasetInfo.newBuilder(dataset).build());
    processedTripsTable = TableId.of(dataset, PROCESSED_TRIPS_TABLE);
    mlDatasetTable = TableId.of(dataset, ML_DATASET_TABLE);
  }

  @Override
  public void close() {
    RemoteBigQueryHelper.forceDelete(bigquery, dataset);
  }

  public void createTableProcessedTrips() {
    createTable(processedTripsTable, PROCESSED_TRIPS_FIELDS);
  }

  public void createTableMlDataset() {
    createTable(mlDatasetTable, ML_DATASET_FIELDS);
  }

  private void createTable(TableId tableId, Iterable<Field> fields) {
    Schema schema = Schema.of(fields);
    TableInfo tableInfo = TableInfo.newBuilder(tableId, StandardTableDefinition.of(schema)).build();
    bigquery.create(tableInfo);
  }

  public boolean insertProcessedTrips(List<List<String>> values)
      throws IOException, InterruptedException {
    return insert(processedTripsTable, values);
  }

  private boolean insert(TableId tableId, List<List<String>> values)
      throws IOException, InterruptedException {
    WriteChannelConfiguration writeChannelConfiguration =
        WriteChannelConfiguration
            .newBuilder(tableId)
            .setFormatOptions(FormatOptions.csv())
            .build();

    TableDataWriteChannel writer = bigquery.writer(writeChannelConfiguration);
    try {
      writer.write(ByteBuffer.wrap(createCsv(values).getBytes(Charsets.UTF_8)));
    } finally {
      writer.close();
    }
    Job job = writer.getJob();
    job = job.waitFor();
    LoadStatistics stats = job.getStatistics();
    return stats.getOutputRows() == 1;
  }

  private String createCsv(List<List<String>> values) {
    StringBuilder csvBuilder = new StringBuilder();
    for (List<String> row : values) {
      csvBuilder.append(Joiner.on(',').join(row)).append('\n');
    }
    return csvBuilder.toString();
  }

  public List<FieldValueList> query(String query) throws InterruptedException {
    TableResult tableResult = bigquery.query(QueryJobConfiguration.newBuilder(query).build());
    List<FieldValueList> results = new ArrayList<>();
    tableResult.getValues().forEach(v -> results.add(v));
    return results;
  }
}
