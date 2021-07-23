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
import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;
import org.apache.avro.generic.GenericRecord;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO.TypedRead.Method;
import org.apache.beam.sdk.io.gcp.bigquery.SchemaAndRecord;
import org.apache.beam.sdk.io.gcp.bigquery.WriteResult;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.options.ValueProvider;
import org.apache.beam.sdk.transforms.Count;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.Mean;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.transforms.SimpleFunction;
import org.apache.beam.sdk.transforms.Wait;
import org.apache.beam.sdk.transforms.WithKeys;
import org.apache.beam.sdk.transforms.join.CoGbkResult;
import org.apache.beam.sdk.transforms.join.CoGroupByKey;
import org.apache.beam.sdk.transforms.join.KeyedPCollectionTuple;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.TupleTag;
import org.apache.beam.sdk.values.TypeDescriptors;
import org.apache.commons.lang.StringEscapeUtils;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Apache Beam pipeline.
 * Retrieves data from the public BigQuery table `bigquery-public-data.chicago_taxi_trips.taxi_trips`,
 * and saves it into the Cloud Storage bucket. The subsequent executions will only read
 * new records.
 */
public class TripsPublicBqToStorage {

    private static final String DST_KEY_COLUMN = "unique_key";
    private static final String DST_TIMESTAMP_COLUMN = "processed_timestamp";

    private static final TableSchema TAXI_ID_SCHEMA = new TableSchema().setFields(List.of(
            new TableFieldSchema().setName(DST_KEY_COLUMN).setType("STRING").setMode("REQUIRED"),
            new TableFieldSchema().setName(DST_TIMESTAMP_COLUMN).setType("TIMESTAMP").setMode("REQUIRED")));

    private static String CSV_HEADER =  "pickup_community_area," +
                                        "day_of_week," +
                                        "is_us_holiday," +
                                        "month," +
                                        "hour_of_day," +
                                        "am_pm," +
                                        "avg_fare_per_trip," +
                                        "number_of_trips";

    public static void main(String[] args) {
        Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(Options.class);

        Pipeline p = Pipeline.create(options);

        PCollection<Trip> trips = p.apply(
                        "Reading from BQ",
                        BigQueryIO.read(new TableRowToTripConverter())
                        .fromQuery(makeQuery(options))
                        .usingStandardSql()
                        .withMethod(Method.DIRECT_READ)
                        .withTemplateCompatibility()
                        .withoutValidation());

        WriteResult taxiIdInsertsResult = trips.apply(
                "Writing to BQ",
                BigQueryIO.<Trip>write()
                        .withFormatFunction(new TripToTaxiIdTableRowConverter())
                        .to(StringEscapeUtils.escapeSql(options.getDestinationTable().get()))
                        .withSchema(TAXI_ID_SCHEMA)
                        .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_NEVER)
                        .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND)
                        .withMethod(BigQueryIO.Write.Method.DEFAULT));

        PCollection<KV<String, Trip>> tripsByKey = trips.apply(
                "Grouping an elements",
                WithKeys.of(TripsPublicBqToStorage::makeAreaHourGroupingKey).withKeyType(TypeDescriptors.strings()));

        PCollection<KV<String, Double>> averageFares = tripsByKey
                .apply("Grouping a fares",
                        MapElements.via(
                                new SimpleFunction<KV<String, Trip>, KV<String, Float>>() {
                                    @Override
                                    public KV<String, Float> apply(KV<String, Trip> r) {
                                        return KV.of(r.getKey(), r.getValue().getFare());
                                    }
                                }))
                .apply("Calculating an average fare", Mean.perKey());

        PCollection<KV<String, Long>> counts = tripsByKey.apply("Counting a number of trips", Count.perKey());

        final TupleTag<Trip> tripsTag = new TupleTag<>();
        final TupleTag<Long> countsTag = new TupleTag<>();
        final TupleTag<Double> averagesTag = new TupleTag<>();

        PCollection<KV<String, CoGbkResult>> joined = KeyedPCollectionTuple
                .of(tripsTag, tripsByKey)
                .and(countsTag, counts)
                .and(averagesTag, averageFares)
                .apply("Joining data", CoGroupByKey.create());

        PCollection<String> csv = joined.apply(
                "Converting to CSV format",
                MapElements.via(new KvToCsvConverter(tripsTag, countsTag, averagesTag)));

        csv.apply("Waiting BQ inserts to finish",
                Wait.on(taxiIdInsertsResult.getFailedInserts()))
            .apply("Saving CSV files",
                TextIO.write()
                    .to(String.format("%s/trips", options.getOutputDirectory()))
                    .withHeader(CSV_HEADER)
                    .withSuffix(".csv")
                    .withoutSharding());

        p.run();
    }

    @VisibleForTesting
    static double roundAverageFare(double avg) {
        return Math.round(avg * 100) / 100d;
    }

    /**
     * Creates BigQuery's TableRow object for inserting into the table taxi_id.
     */
    private static class TripToTaxiIdTableRowConverter implements SerializableFunction<Trip, TableRow> {

        private static final String NOW = Instant.now().toString();

        @Override
        public TableRow apply(Trip trip) {
            return new TableRow()
                    .set(DST_KEY_COLUMN, trip.getUniqueKey())
                    .set(DST_TIMESTAMP_COLUMN, NOW);
        }
    }

    /**
     * Converts a grouped data object into a CSV line.
     */
    @VisibleForTesting
    @AllArgsConstructor
    static class KvToCsvConverter extends SimpleFunction<KV<String, CoGbkResult>, String> {
        final TupleTag<Trip> tripsTag;
        final TupleTag<Long> countsTag;
        final TupleTag<Double> averagesTag;

        private static final DateTimeFormatter amPmFormatter = DateTimeFormatter.ofPattern("a");
        private static final DateTimeFormatter hourOfDayFormatter = DateTimeFormatter.ofPattern("h");

        @Override
        public String apply(KV<String, CoGbkResult> e) {
            CoGbkResult result = e.getValue();
            long numberOfTrips = result.getOnly(countsTag);
            double averageFare = result.getOnly(averagesTag);
            Trip sampleTrip = result.getAll(tripsTag).iterator().next();

            return String.format("%s,%s,%s,%s,%s,%s,%s,%s",
                    sampleTrip.getPickupArea(),
                    sampleTrip.getTripStartHour().getDayOfWeek().getValue(),
                    sampleTrip.isUsHoliday(),
                    sampleTrip.getTripStartHour().getMonthValue(),
                    Integer.valueOf(sampleTrip.getTripStartHour().format(hourOfDayFormatter)),
                    sampleTrip.getTripStartHour().format(amPmFormatter),
                    averageFare,
                    numberOfTrips);
        }
    }

    /**
     * Converts BigQuery's TableRow object into a Trip object.
     */
    @VisibleForTesting
    static class TableRowToTripConverter implements SerializableFunction<SchemaAndRecord, Trip> {
        @Override
        public Trip apply(SchemaAndRecord schemaAndRecord) {
            GenericRecord r = schemaAndRecord.getRecord();
            Trip trip = new Trip(String.valueOf(r.get("unique_key")));
            trip.setTripStartHour(LocalDateTime.parse(String.valueOf(r.get("trip_start_hour"))));
            trip.setPickupArea(Integer.valueOf(String.valueOf(r.get("pickup_community_area"))));
            trip.setPickupLatitude(Double.valueOf(String.valueOf(r.get("pickup_latitude"))));
            trip.setPickupLongitude(Double.valueOf(String.valueOf(r.get("pickup_longitude"))));
            trip.setFare(Float.valueOf(String.valueOf(r.get("fare"))));
            trip.setUsHoliday(Boolean.valueOf(String.valueOf(r.get("is_us_holiday"))));
            return trip;
        }
    }

    /**
     * Creates a grouping key by the pickup area and pickup hour.
     */
    @VisibleForTesting
    static String makeAreaHourGroupingKey(Trip trip) {
        return String.format("%d_%s", trip.getPickupArea(), trip.getTripStartHour().toString());
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
        @Default.String("chicago_taxi_ml_demo_1.taxi_trips_view")
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
        return String.format(
                "SELECT t.unique_key, " +
                    "DATETIME(TIMESTAMP_TRUNC(t.trip_start_timestamp, HOUR)) AS trip_start_hour, " +
                    "t.pickup_latitude, " +
                    "t.pickup_longitude, " +
                    "t.pickup_community_area, " +
                    "IF (h.countryRegionCode IS NULL, false, true) AS is_us_holiday, " +
                    "t.fare " +
                "FROM %s t " +
                "LEFT JOIN chicago_taxi_ml_demo_1.national_holidays h " +
                    "ON TIMESTAMP_TRUNC(t.trip_start_timestamp, DAY) = TIMESTAMP_TRUNC(h.date, DAY) " +
                    "AND h.countryRegionCode = 'US' " +
                "WHERE trip_start_timestamp > TIMESTAMP_SUB(current_timestamp(), INTERVAL %d DAY) " +
                "AND t.trip_start_timestamp IS NOT NULL " +
                "AND t.trip_end_timestamp IS NOT NULL " +
                "AND t.pickup_community_area IS NOT NULL " +
                "AND t.fare IS NOT NULL " +
                "AND t.trip_start_timestamp < t.trip_end_timestamp " +
                "AND t.unique_key NOT IN " +
                    "(SELECT unique_key FROM %s) " +
                "AND (SELECT ST_COVERS(b.boundaries, ST_GEOGPOINT(t.pickup_longitude, t.pickup_latitude)) " +
                    "FROM `chicago_taxi_ml_demo_1.chicago_boundaries` b) = True " +
                "LIMIT %d",
                StringEscapeUtils.escapeSql(options.getSourceTable().get()),
                options.getInterval().get(),
                StringEscapeUtils.escapeSql(options.getDestinationTable().get()),
                options.getLimit().get()
        );
    }
}
