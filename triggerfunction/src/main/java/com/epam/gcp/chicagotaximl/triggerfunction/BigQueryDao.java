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
import com.google.common.annotations.VisibleForTesting;
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

  private static final String ML_DATASET_STATE_QUERY =
      "SELECT area FROM `%1$s.ml_dataset` LIMIT 1";

  @VisibleForTesting
  public static final String NEW_TRIPS_QUERY =
           "INSERT INTO `%1$s.processed_trips` (unique_key) "
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
          + "   AND trip_start_timestamp >= TIMESTAMP('%3$s 00:00:00+00')); ";

  @VisibleForTesting
  public static final String NEW_MODIFICATION_TIMESTAMP_QUERY =
      "INSERT INTO `%1$s.source_modification_timestamps` (modification_timestamp) "
      + "   (SELECT TIMESTAMP_MILLIS(last_modified_time) "
      + "   FROM `%1$s.public_dataset_tables` "
      + "   WHERE table_id = 'taxi_trips' "
      + "   LIMIT 1); ";

  private static final String ML_DATASET_QUERY =
      // temp function returns number of days in a corresponding month for a given date
      "CREATE TEMP FUNCTION n_days_in_month(udate TIMESTAMP) AS ( "
          + "  32 - EXTRACT(DAY FROM TIMESTAMP_ADD(TIMESTAMP_TRUNC(udate, MONTH), INTERVAL 31 DAY)) "
          + "); "
          + ""
          + "INSERT INTO `%1$s.ml_dataset` ("
          + "  WITH tmp_taxi_aggregate AS ( "
          + "    SELECT "
          + "      pickup_community_area AS area, "
          + "      TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS trip_datehour, "
          + "      EXTRACT(DAYOFWEEK FROM trip_start_timestamp) - 1 AS day_of_week, "
          + "      COUNT(*) AS n_trips "
          + "    FROM `%1$s.%2$s` "
          + "    WHERE "
          + "      unique_key IN (SELECT unique_key "
          + "                     FROM `%1$s.processed_trips` WHERE processed_timestamp IS NULL) "
          + "    GROUP BY "
          + "      area, trip_datehour, day_of_week "
          + "    ), "
          + ""
          + "    tmp_taxi_aggregate_extracted AS ( "
          + "      SELECT  "
          + "        area, "
          + "        n_trips, "
          + "        trip_datehour, "
          + "        day_of_week, "
          + "        EXTRACT(HOUR FROM trip_datehour) AS hour, "
          + "        EXTRACT(DAY FROM trip_datehour) AS day, "
          + "        EXTRACT(WEEK(MONDAY) FROM trip_datehour) AS week, "
          + "        EXTRACT(MONTH FROM trip_datehour) AS month, "
          + "        EXTRACT(QUARTER FROM trip_datehour) AS quarter, "
          + "        n_days_in_month(trip_datehour) AS n_days_in_month "
          + "      FROM tmp_taxi_aggregate"
          + "      WHERE MOD(ABS(FARM_FINGERPRINT(CAST(DATE(trip_datehour) AS STRING))), 10) < 8), "
          + ""
          + "    tmp_taxi_aggregate_scaled AS ( "
          + "      SELECT "
          + "        area, "
          + "        n_trips, "
          + "        n_trips + (RAND() - .5) * POWER(2, -16) AS n_trips_num, "
          + "        trip_datehour,  "
          + "        hour, "
          + "        hour / 24 AS hour_num, "
          + "        day, "
          + "        (day - 1) / n_days_in_month AS day_num, "
          + "        week, "
          + "        week / 53 AS week_num, "
          + "        month, "
          + "        (month - 1) / 12 AS month_num, "
          + "        quarter, "
          + "        (quarter - 1) / 4 AS quarter_num, "
          + "        day_of_week, "
          + "        (day_of_week - 1) / 7 AS day_of_week_num, "
          + "        (day_of_week - 1 + hour / 24 ) / 7 AS weekday_hour_num, "
          + "        ( "
          + "          month-1 + (day - 1 + hour / 24) / n_days_in_month "
          + "        ) / 12 AS yearday_hour_num "
          + "      FROM tmp_taxi_aggregate_extracted "
          + "    ), "
          + ""
          + "    tmp_taxi_result AS ( "
          + "      SELECT "
          + "        area,  "
          + "        EXTRACT(YEAR FROM trip_datehour) AS year,  "
          + "        quarter,  "
          + "        quarter_num, "
          + "        COS(quarter_num * 2*ACOS(-1)) AS quarter_cos, "
          + "        SIN(quarter_num * 2*ACOS(-1)) AS quarter_sin, "
          + "        month,  "
          + "        month_num, "
          + "        COS(month_num * 2*ACOS(-1)) AS month_cos, "
          + "        SIN(month_num * 2*ACOS(-1)) AS month_sin, "
          + "        day,  "
          + "        day_num,  "
          + "        COS(day_num * 2*ACOS(-1)) AS day_cos, "
          + "        SIN(day_num * 2*ACOS(-1)) AS day_sin, "
          + "        hour, "
          + "        hour_num, "
          + "        COS(hour_num * 2*ACOS(-1)) AS hour_cos, "
          + "        SIN(hour_num * 2*ACOS(-1)) AS hour_sin, "
          + "        IF (hour < 12, 'am', 'pm') AS day_period, "
          + "        week,  "
          + "        week_num, "
          + "        COS(week_num * 2*ACOS(-1)) AS week_cos, "
          + "        SIN(week_num * 2*ACOS(-1)) AS week_sin, "
          + "        day_of_week, "
          + "        day_of_week_num, "
          + "        COS(day_of_week_num * 2*ACOS(-1)) AS day_of_week_cos, "
          + "        SIN(day_of_week_num * 2*ACOS(-1)) AS day_of_week_sin, "
          + "        weekday_hour_num, "
          + "        COS(weekday_hour_num * 2*ACOS(-1)) AS weekday_hour_cos, "
          + "        SIN(weekday_hour_num * 2*ACOS(-1)) AS weekday_hour_sin, "
          + "        yearday_hour_num, "
          + "        COS(yearday_hour_num * 2*ACOS(-1)) AS yearday_hour_cos, "
          + "        SIN(yearday_hour_num * 2*ACOS(-1)) AS yearday_hour_sin, "
          + "        day_of_week IN (6, 7) AS is_weekend, "
          + "        IF (h.holidayName IS NULL, False, True) AS is_holiday, "
          + "        n_trips, "
          + "        n_trips_num, "
          + "        LOG(n_trips_num + 1) AS log_n_trips, "
          + "        CASE  "
          + "          WHEN n_trips < 10 THEN 1  " // 'bucket_[0-10)'
          + "          WHEN n_trips >= 10 AND n_trips < 15 THEN 2 " // 'bucket_[10-15)'
          + "          WHEN n_trips >= 15 AND n_trips < 20 THEN 3 " // 'bucket_[15-20)'
          + "          WHEN n_trips >= 20 AND n_trips < 30 THEN 5 " // 'bucket_[20-30)'
          + "          WHEN n_trips >= 30 AND n_trips < 50 THEN 7 " // 'bucket_[30-50)'
          + "          WHEN n_trips >= 50 THEN 10 " // 'bucket_[100-)'
          + "        END AS trips_bucket "
          + "      FROM  "
          + "        tmp_taxi_aggregate_scaled t "
          + "      LEFT JOIN `%1$s.national_holidays` h "
          + "      ON TIMESTAMP_TRUNC(t.trip_datehour, DAY) = h.date) "
          + ""
          + "  SELECT "
          + "    *, "
          + "    trips_bucket + (RAND() - .5) * POWER(2, -16) AS trips_bucket_num "
          + "  FROM tmp_taxi_result "
          + "  WHERE trips_bucket <> 1 "
          + "  OR RAND() < ("
          + "      SELECT COUNTIF(trips_bucket != 1) / COUNTIF(trips_bucket = 1) / (COUNT(DISTINCT trips_bucket) - 1)"
          + "      FROM tmp_taxi_result"
          + "    ) "
          + "); ";

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
   * Returns False if the state of 'ml_dataset' table is invalid.
   */
  public boolean verifyMlDatasetState() throws InterruptedException {
    Builder queryBuilder = QueryJobConfiguration.newBuilder(
        String.format(ML_DATASET_STATE_QUERY, StringEscapeUtils.escapeSql(dataset)));
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
        String.format(
            "BEGIN TRANSACTION; "
                + NEW_TRIPS_QUERY
                + NEW_MODIFICATION_TIMESTAMP_QUERY
                + ML_DATASET_QUERY
                + "COMMIT TRANSACTION;",
            StringEscapeUtils.escapeSql(dataset),
            StringEscapeUtils.escapeSql(sourceTable),
            startDate.toString()
        ));
    bigquery.query(queryBuilder.build());
  }
}
