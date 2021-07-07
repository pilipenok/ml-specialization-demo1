/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */
package com.epam.gcp.chicagotaximl.dataflow;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.PipelineResult;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO.TypedRead;
import org.apache.beam.sdk.options.*;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.TypeDescriptors;
import org.apache.commons.lang.StringEscapeUtils;

import java.util.List;

/**
 * Apache Beam pipeline.
 * Retrieves data from the public BigQuery table `bigquery-public-data.chicago_taxi_trips.taxi_trips`,
 * and saves it into the Cloud Storage bucket. The subsequent executions will only read
 * new records.
 */
public class TripsPublicBqToStorage {

    private static final String SRC_KEY_COLUMN = "unique_key";
    private static final String SRC_PICKUP_LATITUDE_COLUMN = "pickup_latitude";
    private static final String DST_KEY_COLUMN = "unique_key";

    private static final TableSchema DST_SCHEMA = new TableSchema().setFields(List.of(
                    new TableFieldSchema().setName(DST_KEY_COLUMN).setType("STRING").setMode("REQUIRED")));

    public static void main(String[] args) {
        Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(Options.class);

        Pipeline p = Pipeline.create(options);

        PCollection<TableRow> rows = p.apply("Reading from BigQuery",
                    BigQueryIO.readTableRows()
                    .fromQuery(makeQuery(options))
                    .usingStandardSql()
                    .withMethod(TypedRead.Method.DIRECT_READ)
                    .withTemplateCompatibility()
                    .withoutValidation());

        rows.apply("Writing to BigQuery",
                BigQueryIO.<TableRow>write()
                        .withFormatFunction(v -> (new TableRow().set(DST_KEY_COLUMN, v.get(SRC_KEY_COLUMN))))
                        .to(StringEscapeUtils.escapeSql(options.getDestinationTable().get()))
                        .withSchema(DST_SCHEMA)
                        .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_NEVER)
                        .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND)
                        .withMethod(BigQueryIO.Write.Method.DEFAULT));

        PCollection<String> csv = rows.apply("Converting to CSV format",
                MapElements.into(TypeDescriptors.strings())
                        .via(v -> (String.format("%s,%s", v.get(SRC_KEY_COLUMN), v.get(SRC_PICKUP_LATITUDE_COLUMN)))));

        csv.apply("Saving CSV files",
                TextIO.write()
                        .to(String.format("%s/trips", options.getOutputDirectory()))
                        .withSuffix(".csv"));

        PipelineResult run = p.run();

        if (!options.getRunner().equals(PipelineOptions.DirectRunner.class)) {
            // waitUntilFinish() is not supported by local runner
            run.waitUntilFinish();
        }
    }

    /**
     * Pipeline options.
     */
    public interface Options extends PipelineOptions {

        @Description("Location to store output CSV files (without trailing '/').")
        @Default.String("gs://chicago-taxi-ml-demo-1/trips")
        ValueProvider<String> getOutputDirectory();

        void setOutputDirectory(ValueProvider<String> outputDirectory);


        @Description("The BigQuery source table.")
        @Default.String("bigquery-public-data.chicago_taxi_trips.taxi_trips")
        ValueProvider<String> getSourceTable();

        void setSourceTable(ValueProvider<String> sourceTable);


        @Description("BigQuery table that contains IDs of the processed trips.")
        @Default.String("chicago_taxi_ml_demo_1.taxi_id")
        ValueProvider<String> getDestinationTable();

        void setDestinationTable(ValueProvider<String> destinationTable);


        @Description("The period of time in days, for which the trips should be fetched from the source table.")
        @Default.Integer(1100)
        ValueProvider<Integer> getInterval();

        void setInterval(ValueProvider<Integer> interval);


        @Description("Query limit.")
        @Default.Integer(2000000000)
        ValueProvider<Integer> getLimit();

        void setLimit(ValueProvider<Integer> limit);
    }

    private static String makeQuery(Options options) {
        return String.format("SELECT unique_key, pickup_latitude " +
                                "FROM %s " +
                                "WHERE trip_start_timestamp > TIMESTAMP_SUB(current_timestamp(), INTERVAL %d DAY) " +
                                "AND unique_key NOT IN " +
                                "   (SELECT unique_key FROM %s) " +
                                "LIMIT %d",
                StringEscapeUtils.escapeSql(options.getSourceTable().get()),
                options.getInterval().get(),
                StringEscapeUtils.escapeSql(options.getDestinationTable().get()),
                options.getLimit().get());
    }
}
