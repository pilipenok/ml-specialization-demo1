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
  }

  private static String makeQuery(Options options) {
    return makeQuery(options.getDataset().get());
  }

  private static String makeQuery(String dataset) {
    return String.format(
        "SELECT * FROM `%1$s.ml_dataset`",
        StringEscapeUtils.escapeSql(dataset));
  }
}
