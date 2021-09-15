/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.QueryJobConfiguration.Builder;
import com.google.cloud.bigquery.TableResult;
import java.time.LocalDate;
import org.apache.commons.lang.StringEscapeUtils;
import org.springframework.beans.factory.annotation.Autowired;

/**
 * BigQuery DAO.
 */
public class BigQueryDao {

  @Autowired
  private BigQuery bigquery;

  private final String dataset;
  private final String sourceTable;
  private final LocalDate startDate;

  // The query returns 1 row if table `taxi_trips` has been updated since last pipeline execution or
  // that is first pipeline run.
  private static final String MODIFIED_TIME_QUERY =
      "SELECT last_modified_time "
          + "FROM `%1$s.public_dataset_tables` "
          + "WHERE table_id = 'taxi_trips' "
          + "AND (TIMESTAMP_MILLIS(last_modified_time) > "
          + "    (SELECT MAX(modification_timestamp) FROM `%1$s.source_modification_timestamps`) "
          + "  OR (SELECT modification_timestamp "
          + "     FROM `%1$s.source_modification_timestamps` LIMIT 1) IS NULL "
          + ") "
          + "LIMIT 1";

  private static final String PROCESSED_STATE_QUERY =
      "SELECT unique_key "
          + "FROM `%1$s.processed_trips` "
          + "WHERE processed_timestamp IS NULL "
          + "LIMIT 1";

  private static final String NEW_TRIPS_QUERY =
      "BEGIN TRANSACTION; "
          + ""
          + "INSERT INTO `%1$s.processed_trips` (unique_key) "
          + "   (SELECT unique_key "
          + "   FROM `%1$s.%2$s` "
          + "   WHERE unique_key NOT IN "
          + "         (SELECT unique_key FROM `%1$s.processed_trips` "
          + "         WHERE processed_timestamp IS NOT NULL) "
          + "   AND taxi_id IS NOT NULL "
          + "   AND trip_start_timestamp IS NOT NULL "
          + "   AND trip_end_timestamp IS NOT NULL "
          + "   AND trip_seconds > 300 "
          + "   AND trip_miles IS NOT NULL "
          + "   AND fare IS NOT NULL "
          + "   AND trip_total IS NOT NULL "
          + "   AND payment_type IS NOT NULL "
          + "   AND pickup_community_area IS NOT NULL  "
          + "   AND pickup_latitude IS NOT NULL  "
          + "   AND pickup_longitude IS NOT NULL "
          + "   AND ST_COVERS( "
          + "       (SELECT boundaries FROM `%1$s.chicago_boundaries`), "
          + "       ST_GEOGPOINT(pickup_longitude, pickup_latitude)) "
          + "   AND ( "
          + "       dropoff_community_area IS NOT NULL "
          + "       OR (dropoff_latitude IS NOT NULL AND dropoff_longitude IS NOT NULL) "
          + "   ) "
          + "   AND trip_start_timestamp >= TIMESTAMP('%3$s 00:00:00+00')); "
          + ""
          + "INSERT INTO `%1$s.source_modification_timestamps` (modification_timestamp) "
          + "   (SELECT TIMESTAMP_MILLIS(last_modified_time) "
          + "   FROM `%1$s.public_dataset_tables` "
          + "   WHERE table_id = 'taxi_trips' "
          + "   LIMIT 1); "
          + ""
          + "COMMIT TRANSACTION;";

  /**
   * The BigQuery DAO object.
   *
   * @param dataset The BigQuery dataset name.
   * @param sourceTable Taxi trips table name.
   * @param startDate The earliest trips to evaluate.
   */
  public BigQueryDao(String dataset, String sourceTable, LocalDate startDate) {
    this.dataset = dataset;
    this.sourceTable = sourceTable;
    this.startDate = startDate;
  }

  /**
   * Returns True if BigQuery table 'bigquery-public-data.chicago_taxi_trips.taxi_trips'
   * has been modified since previous Dataflow run.
   */
  public boolean checkLastModifiedTime() throws InterruptedException {
    Builder queryBuilder = QueryJobConfiguration.newBuilder(
        String.format(MODIFIED_TIME_QUERY, StringEscapeUtils.escapeSql(dataset)));
    TableResult results = bigquery.query(queryBuilder.build());
    return results.getTotalRows() > 0;
  }

  /**
   * Returns False if the state of 'processed_trips' table is invalid.
   */
  public boolean verifyPreprocessedState() throws InterruptedException {
    Builder queryBuilder = QueryJobConfiguration.newBuilder(
        String.format(PROCESSED_STATE_QUERY, StringEscapeUtils.escapeSql(dataset)));
    TableResult results = bigquery.query(queryBuilder.build());
    return results.getTotalRows() == 0;
  }

  /**
   * Copies IDs of new trips from table 'taxi_trips' to 'processed_trips'.
   * Inserts the table modification timestamp of the table 'taxi_trips' into the
   * 'source_modification_timestamps' table.
   */
  public void prepareTripsForDataflow() throws InterruptedException {
    Builder queryBuilder = QueryJobConfiguration.newBuilder(
        String.format(NEW_TRIPS_QUERY,
            StringEscapeUtils.escapeSql(dataset),
            StringEscapeUtils.escapeSql(sourceTable),
            startDate.toString()
        ));
    bigquery.query(queryBuilder.build());
  }
}
