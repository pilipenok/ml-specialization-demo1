/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction.integration;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.CsvOptions;
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
import com.google.common.io.Resources;
import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import lombok.Getter;

/**
 * Helper class for integration testing with BigQuery.
 */
public class BigQueryTesting implements Closeable {

  static final String PROCESSED_TRIPS_TABLE = "processed_trips";
  static final String TAXI_TRIPS_TABLE = "taxi_trips";
  static final String PUBLIC_TABLES_TABLE = "public_dataset_tables";
  static final String SOURCE_MODIFICATION_TABLE = "source_modification_timestamps";
  static final String CHICAGO_BOUNDARIES_RAW_TABLE = "chicago_boundaries_raw";
  static final String CHICAGO_BOUNDARIES_TABLE = "chicago_boundaries";

  private final RemoteBigQueryHelper bigqueryHelper = RemoteBigQueryHelper.create();

  @Getter
  private final BigQuery bigquery = bigqueryHelper.getOptions().getService();

  @Getter
  private final String dataset = RemoteBigQueryHelper.generateDatasetName();

  private static final List<Field> TAXI_TRIPS_FIELDS = Arrays.asList(
      Field.of("unique_key", StandardSQLTypeName.STRING),
      Field.of("taxi_id", StandardSQLTypeName.STRING),
      Field.of("trip_start_timestamp", StandardSQLTypeName.TIMESTAMP),
      Field.of("trip_end_timestamp", StandardSQLTypeName.TIMESTAMP),
      Field.of("trip_seconds", StandardSQLTypeName.INT64),
      Field.of("trip_miles", StandardSQLTypeName.FLOAT64),
      Field.of("pickup_community_area", StandardSQLTypeName.INT64),
      Field.of("dropoff_community_area", StandardSQLTypeName.INT64),
      Field.of("fare", StandardSQLTypeName.FLOAT64),
      Field.of("trip_total", StandardSQLTypeName.FLOAT64),
      Field.of("payment_type", StandardSQLTypeName.STRING),
      Field.of("pickup_latitude", StandardSQLTypeName.FLOAT64),
      Field.of("pickup_longitude", StandardSQLTypeName.FLOAT64),
      Field.of("dropoff_latitude", StandardSQLTypeName.FLOAT64),
      Field.of("dropoff_longitude", StandardSQLTypeName.FLOAT64));

  private static final List<Field> PROCESSED_TRIPS_FIELDS = Arrays.asList(
      Field.of("unique_key", StandardSQLTypeName.STRING),
      Field.of("processed_timestamp", StandardSQLTypeName.TIMESTAMP));

  private static final List<Field> PUBLIC_TABLES_FIELDS = Arrays.asList(
      Field.of("table_id", StandardSQLTypeName.STRING),
      Field.of("last_modified_time", StandardSQLTypeName.INT64));

  private static final List<Field> SOURCE_MODIFICATION_FIELDS = Arrays.asList(
      Field.of("modification_timestamp", StandardSQLTypeName.TIMESTAMP));

  private static final List<Field> CHICAGO_BOUNDARIES_RAW_FIELDS = Arrays.asList(
      Field.of("col1", StandardSQLTypeName.STRING),
      Field.of("boundaries", StandardSQLTypeName.STRING),
      Field.of("col3", StandardSQLTypeName.STRING),
      Field.of("col4", StandardSQLTypeName.STRING),
      Field.of("col5", StandardSQLTypeName.STRING));

  private static final List<Field> CHICAGO_BOUNDARIES_FIELDS = Arrays.asList(
      Field.of("boundaries", StandardSQLTypeName.GEOGRAPHY));

  private TableId taxiTripsTable;
  private TableId processedTripsTable;
  private TableId publicTablesTable;
  private TableId sourceModificationTable;
  private TableId chicagoBoundariesRawTable;
  private TableId chicagoBoundariesTable;

  public BigQueryTesting() {
    bigquery.create(DatasetInfo.newBuilder(dataset).build());
    taxiTripsTable = TableId.of(dataset, TAXI_TRIPS_TABLE);
    processedTripsTable = TableId.of(dataset, PROCESSED_TRIPS_TABLE);
    publicTablesTable = TableId.of(dataset, PUBLIC_TABLES_TABLE);
    sourceModificationTable = TableId.of(dataset, SOURCE_MODIFICATION_TABLE);
    chicagoBoundariesRawTable = TableId.of(dataset, CHICAGO_BOUNDARIES_RAW_TABLE);
    chicagoBoundariesTable = TableId.of(dataset, CHICAGO_BOUNDARIES_TABLE);
  }

  @Override
  public void close() {
    RemoteBigQueryHelper.forceDelete(bigquery, dataset);
  }

  public void createTableProcessedTrips() {
    createTable(processedTripsTable, PROCESSED_TRIPS_FIELDS);
  }

  public void createTableTaxiTrips() {
    createTable(taxiTripsTable, TAXI_TRIPS_FIELDS);
  }

  public void createTablePublicDatasetTables() {
    createTable(publicTablesTable, PUBLIC_TABLES_FIELDS);
  }

  public void createTableSourceModificationTimestamps() {
    createTable(sourceModificationTable, SOURCE_MODIFICATION_FIELDS);
  }

  public void createTableChicagoBoundaries() {
    createTable(chicagoBoundariesTable, CHICAGO_BOUNDARIES_FIELDS);
    createTable(chicagoBoundariesRawTable, CHICAGO_BOUNDARIES_RAW_FIELDS);
  }

  private void createTable(TableId tableId, Iterable<Field> fields) {
    Schema schema = Schema.of(fields);
    TableInfo tableInfo = TableInfo.newBuilder(tableId, StandardTableDefinition.of(schema)).build();
    bigquery.create(tableInfo);
  }

  public boolean insertTaxiTrips(List<List<String>> values) throws Exception {
    return insert(taxiTripsTable, values);
  }

  public boolean insertProcessedTrips(List<List<String>> values) throws Exception {
    return insert(processedTripsTable, values);
  }

  public boolean insertPublicDatasetTables(List<List<String>> values) throws Exception {
    return insert(publicTablesTable, values);
  }

  public boolean insertSourceModificationTimestamps(List<List<String>> values) throws Exception {
    return insert(sourceModificationTable, values);
  }

  private boolean insert(TableId tableId, List<List<String>> values) throws Exception {
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

  private static String createCsv(List<List<String>> values) {
    StringBuilder csvBuilder = new StringBuilder();
    for (List<String> row : values) {
      csvBuilder.append(Joiner.on(',').join(row)).append('\n');
    }
    return csvBuilder.toString();
  }

  public List<FieldValueList> query(String query) throws Exception {
    TableResult tableResult = bigquery.query(QueryJobConfiguration.newBuilder(query).build());
    List<FieldValueList> results = new ArrayList<>();
    tableResult.getValues().forEach(v -> results.add(v));
    return results;
  }

  public void fillTableChicagoBoundaries() throws Exception {
    String boundaries = Resources.toString(Resources.getResource("City_Boundary.csv"), Charsets.UTF_8);
    WriteChannelConfiguration writeChannelConfiguration =
        WriteChannelConfiguration
            .newBuilder(chicagoBoundariesRawTable)
            .setFormatOptions(CsvOptions.newBuilder().setSkipLeadingRows(1).build())
            .build();

    TableDataWriteChannel writer = bigquery.writer(writeChannelConfiguration);
    try {
      writer.write(ByteBuffer.wrap(boundaries.getBytes(Charsets.UTF_8)));
    } finally {
      writer.close();
    }
    Job job = writer.getJob();
    job.waitFor();

    query(String.format(
        "INSERT INTO `%1$s.%2$s` (boundaries) "
        + "(SELECT ST_GEOGFROMTEXT(boundaries) FROM `%1$s.%3$s`)",
        dataset, CHICAGO_BOUNDARIES_TABLE, CHICAGO_BOUNDARIES_RAW_TABLE));
  }


  public void truncateTaxiTrips() throws Exception {
    query(String.format("TRUNCATE TABLE `%s.%s`", dataset, TAXI_TRIPS_TABLE));
  }

  public void truncateProcessedTrips() throws Exception {
    query(String.format("TRUNCATE TABLE `%s.%s`", dataset, PROCESSED_TRIPS_TABLE));
  }

  public void truncatePublicDatasetTables() throws Exception {
    query(String.format("TRUNCATE TABLE `%s.%s`", dataset, PUBLIC_TABLES_TABLE));
  }

  public void truncateSourceModificationTimestamps() throws Exception {
    query(String.format("TRUNCATE TABLE `%s.%s`", dataset, SOURCE_MODIFICATION_TABLE));
  }
}
