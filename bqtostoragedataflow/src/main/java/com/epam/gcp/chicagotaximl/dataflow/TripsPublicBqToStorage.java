/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.dataflow;

import com.google.common.annotations.VisibleForTesting;
import org.apache.avro.generic.GenericRecord;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO.TypedRead.Method;
import org.apache.beam.sdk.io.gcp.bigquery.SchemaAndRecord;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.options.ValueProvider;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.values.PCollection;
import org.apache.commons.lang.StringEscapeUtils;

/**
 * Apache Beam pipeline.
 * Retrieves data from a public BigQuery table `bigquery-public-data.chicago_taxi_trips.taxi_trips`,
 * and saves it into a Cloud Storage bucket.
 */
public class TripsPublicBqToStorage {

  private static final String CSV_HEADER =  "area,year,"
      + "quarter,quarter_num,quarter_cos,quarter_sin,"
      + "month,month_num,month_cos,month_sin,"
      + "day,day_num,day_cos,day_sin,"
      + "hour,hour_num,hour_cos,hour_sin,"
      + "day_period,"
      + "week,week_num,week_cos,week_sin,"
      + "day_of_week,day_of_week_num,day_of_week_cos,day_of_week_sin,"
      + "weekday_hour_num,weekday_hour_cos,weekday_hour_sin,"
      + "yearday_hour_num,yearday_hour_cos,yearday_hour_sin,"
      + "is_weekend,is_holiday,n_trips,n_trips_num,log_n_trips,trips_bucket,trips_bucket_num";

  private static final String[] HEADERS = CSV_HEADER.split(",");

  /**
   * An entry point for the Apache Beam pipeline.
   */
  public static void main(String[] args) {
    Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(Options.class);

    Pipeline p = Pipeline.create(options);
    
    PCollection<String> csv = p.apply(
            "Fetching trips",
            BigQueryIO.read(new TableRowToCsv())
                .fromQuery(makeQuery(options))
                .usingStandardSql()
                .withMethod(Method.DIRECT_READ)
                .withTemplateCompatibility()
                .withoutValidation());

    csv.apply("Saving CSV file",
          TextIO.write()
              .to(String.format("%s/trips", options.getOutputDirectory()))
              .withHeader(CSV_HEADER)
              .withSuffix(".csv")
              .withoutSharding());

    p.run();
  }

  /**
   * Converts BigQuery's TableRow object into a CSV row.
   */
  @VisibleForTesting
  static class TableRowToCsv implements SerializableFunction<SchemaAndRecord, String> {
    @Override
    public String apply(SchemaAndRecord schemaAndRecord) {
      GenericRecord r = schemaAndRecord.getRecord();
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < HEADERS.length; i++) {
        sb.append(r.get(HEADERS[i]));
        if (i < HEADERS.length - 1) {
          sb.append(",");
        }
      }
      return sb.toString();
    }
  }

  /**
   * Pipeline options.
   */
  public interface Options extends PipelineOptions {

    @Description("Location to store output CSV files (without trailing '/').")
    ValueProvider<String> getOutputDirectory();

    void setOutputDirectory(ValueProvider<String> outputDirectory);


    @Description("BigQuery dataset.")
    ValueProvider<String> getDataset();

    void setDataset(ValueProvider<String> dataset);


    @Description("BigQuery source table.")
    @Default.String("processed_trips")
    ValueProvider<String> getSourceTable();

    void setSourceTable(ValueProvider<String> sourceTable);
  }

  private static String makeQuery(Options options) {
    return makeQuery(options.getDataset().get(), options.getSourceTable().get());
  }

  private static String makeQuery(String dataset, String sourceTable) {
    return String.format(
        // temp function returns number of days in a corresponding month for a given date
        "CREATE TEMP FUNCTION n_days_in_month(udate TIMESTAMP) AS ( "
            + "  32 - EXTRACT(DAY FROM TIMESTAMP_ADD(TIMESTAMP_TRUNC(udate, MONTH), INTERVAL 31 DAY)) "
            + "); "
            + ""
            + "WITH tmp_taxi_aggregate AS ( "
            + "  SELECT "
            + "    pickup_community_area AS area, "
            + "    TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS trip_datehour, "
            + "    EXTRACT(DAYOFWEEK FROM trip_start_timestamp) - 1 AS day_of_week, "
            + "    COUNT(*) AS n_trips "
            + "  FROM `%1$s.%2$s` "
            + "  WHERE "
            + "    unique_key IN (SELECT unique_key "
            + "                   FROM `%1$s.processed_trips` WHERE processed_timestamp IS NULL) "
            + "  GROUP BY "
            + "    area, trip_datehour, day_of_week "
            + "  ), "
            + ""
            + "  tmp_taxi_aggregate_extracted AS ( "
            + "    SELECT  "
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
            + "    FROM tmp_taxi_aggregate"
            + "    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(DATE(trip_datehour) AS STRING))), 10) < 8), "
            + ""
            + "  tmp_taxi_aggregate_scaled AS ( "
            + "    SELECT "
            + "      area, "
            + "      n_trips, "
            + "      n_trips + (RAND() - .5) * POWER(2, -16) AS n_trips_num, "
            + "      trip_datehour,  "
            + "      hour, "
            + "      hour / 24 AS hour_num, "
            + "      day, "
            + "      (day - 1) / n_days_in_month AS day_num, "
            + "      week, "
            + "      week / 53 AS week_num, "
            + "      month, "
            + "      (month - 1) / 12 AS month_num, "
            + "      quarter, "
            + "      (quarter - 1) / 4 AS quarter_num, "
            + "      day_of_week, "
            + "      (day_of_week - 1) / 7 AS day_of_week_num, "
            + "      (day_of_week - 1 + hour / 24 ) / 7 AS weekday_hour_num, "
            + "      ( "
            + "        month-1 + (day - 1 + hour / 24) / n_days_in_month "
            + "      ) / 12 AS yearday_hour_num "
            + "    FROM tmp_taxi_aggregate_extracted "
            + "  ), "
            + ""
            + "  tmp_taxi_result AS ( "
            + "    SELECT "
            + "      area,  "
            + "      EXTRACT(YEAR FROM trip_datehour) AS year,  "
            + "      quarter,  "
            + "      quarter_num, "
            + "      COS(quarter_num * 2*ACOS(-1)) AS quarter_cos, "
            + "      SIN(quarter_num * 2*ACOS(-1)) AS quarter_sin, "
            + "      month,  "
            + "      month_num, "
            + "      COS(month_num * 2*ACOS(-1)) AS month_cos, "
            + "      SIN(month_num * 2*ACOS(-1)) AS month_sin, "
            + "      day,  "
            + "      day_num,  "
            + "      COS(day_num * 2*ACOS(-1)) AS day_cos, "
            + "      SIN(day_num * 2*ACOS(-1)) AS day_sin, "
            + "      hour, "
            + "      hour_num, "
            + "      COS(hour_num * 2*ACOS(-1)) AS hour_cos, "
            + "      SIN(hour_num * 2*ACOS(-1)) AS hour_sin, "
            + "      IF (hour < 12, 'am', 'pm') AS day_period, "
            + "      week,  "
            + "      week_num, "
            + "      COS(week_num * 2*ACOS(-1)) AS week_cos, "
            + "      SIN(week_num * 2*ACOS(-1)) AS week_sin, "
            + "      day_of_week, "
            + "      day_of_week_num, "
            + "      COS(day_of_week_num * 2*ACOS(-1)) AS day_of_week_cos, "
            + "      SIN(day_of_week_num * 2*ACOS(-1)) AS day_of_week_sin, "
            + "      weekday_hour_num, "
            + "      COS(weekday_hour_num * 2*ACOS(-1)) AS weekday_hour_cos, "
            + "      SIN(weekday_hour_num * 2*ACOS(-1)) AS weekday_hour_sin, "
            + "      yearday_hour_num, "
            + "      COS(yearday_hour_num * 2*ACOS(-1)) AS yearday_hour_cos, "
            + "      SIN(yearday_hour_num * 2*ACOS(-1)) AS yearday_hour_sin, "
            + "      day_of_week IN (6, 7) AS is_weekend, "
            + "      IF (h.holidayName IS NULL, False, True) AS is_holiday, "
            + "      n_trips, "
            + "      n_trips_num, "
            + "      LOG(n_trips_num + 1) AS log_n_trips, "
            + "      CASE  "
            + "        WHEN n_trips < 10 THEN 1  " // 'bucket_[0-10)'
            + "        WHEN n_trips >= 10 AND n_trips < 15 THEN 2 " // 'bucket_[10-15)'
            + "        WHEN n_trips >= 15 AND n_trips < 20 THEN 3 " // 'bucket_[15-20)'
            + "        WHEN n_trips >= 20 AND n_trips < 30 THEN 5 " // 'bucket_[20-30)'
            + "        WHEN n_trips >= 30 AND n_trips < 50 THEN 7 " // 'bucket_[30-50)'
            + "        WHEN n_trips >= 50 THEN 10 " // 'bucket_[100-)'
            + "      END AS trips_bucket "
            + "    FROM  "
            + "      tmp_taxi_aggregate_scaled t "
            + "    LEFT JOIN `%1$s.national_holidays` h "
            + "    ON TIMESTAMP_TRUNC(t.trip_datehour, DAY) = h.date) "
            + ""
            + "SELECT "
            + "  *, "
            + "  trips_bucket + (RAND() - .5) * POWER(2, -16) AS trips_bucket_num "
            + "FROM tmp_taxi_result "
            + "WHERE trips_bucket <> 1 "
            + "OR RAND() < ("
            + "    SELECT COUNTIF(trips_bucket != 1) / COUNTIF(trips_bucket = 1) / (COUNT(DISTINCT trips_bucket) - 1)"
            + "    FROM tmp_taxi_result"
            + "  )",
        StringEscapeUtils.escapeSql(dataset), StringEscapeUtils.escapeSql(sourceTable));
  }
}
