/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.dataflow;

import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.AverageFn.Accum;
import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import com.google.common.annotations.VisibleForTesting;
import java.io.Serializable;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
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
import org.apache.beam.sdk.transforms.Combine;
import org.apache.beam.sdk.transforms.Combine.CombineFn;
import org.apache.beam.sdk.transforms.Count;
import org.apache.beam.sdk.transforms.MapElements;
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

/**
 * Apache Beam pipeline.
 * Retrieves data from a public BigQuery table `bigquery-public-data.chicago_taxi_trips.taxi_trips`,
 * and saves it into a Cloud Storage bucket. The subsequent executions will only read
 * new records.
 */
public class TripsPublicBqToStorage {

  private static final String DST_KEY_COLUMN = "unique_key";
  private static final String DST_TIMESTAMP_COLUMN = "processed_timestamp";

  private static final TableSchema PROCESSED_TRIPS_SCHEMA = new TableSchema().setFields(List.of(
      new TableFieldSchema().setName(DST_KEY_COLUMN).setType("STRING").setMode("REQUIRED"),
      new TableFieldSchema().setName(DST_TIMESTAMP_COLUMN).setType("TIMESTAMP")
          .setMode("REQUIRED")));

  private static String CSV_HEADER =  "pickup_community_area,"
                                        + "day_of_week,"
                                        + "is_us_holiday,"
                                        + "month,"
                                        + "hour_of_day,"
                                        + "am_pm,"
                                        + "avg_fare_per_trip,"
                                        + "number_of_trips";

  /**
   * An entry point for the Apache Beam pipeline.
   */
  public static void main(String[] args) {
    Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(Options.class);

    Pipeline p = Pipeline.create(options);

    PCollection<Trip> trips = p.apply(
            "Fetching trips",
            BigQueryIO.read(new TableRowToTripConverter())
                .fromQuery(makeQuery(options))
                .usingStandardSql()
                .withMethod(Method.DIRECT_READ)
                .withTemplateCompatibility()
                .withoutValidation());

    WriteResult processedTripsInsertResult = trips.apply(
            "Saving IDs",
            BigQueryIO.<Trip>write()
                    .withFormatFunction(new TripToProcessedTripTableRowConverter())
                    .to(StringEscapeUtils.escapeSql(
                        options.getDataset().get()) + ".processed_trips")
                    .withSchema(PROCESSED_TRIPS_SCHEMA)
                    .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_NEVER)
                    .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND)
                    .withMethod(BigQueryIO.Write.Method.DEFAULT));

    PCollection<KV<String, Trip>> tripsByKey = trips.apply(
            "Grouping trips",
            WithKeys.of(TripsPublicBqToStorage::makeAreaHourGroupingKey)
                    .withKeyType(TypeDescriptors.strings()));

    PCollection<KV<String, Double>> averageFares =
        tripsByKey
            .apply("Calculating average fare", Combine.perKey(new AverageFn()));

    PCollection<KV<String, Long>> counts = tripsByKey.apply("Calculating number of trips",
            Count.perKey());

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
            Wait.on(processedTripsInsertResult.getFailedInserts()))
        .apply("Saving CSV file",
          TextIO.write()
              .to(String.format("%s/trips", options.getOutputDirectory()))
              .withHeader(CSV_HEADER)
              .withSuffix(".csv")
              .withoutSharding());

    p.run();
  }

  /**
   * Average function.
   */
  public static class AverageFn extends CombineFn<Trip, Accum, Double>  {

    public static class Accum implements Serializable {
      double sum = 0;
      int count = 0;
    }

    @Override
    public Accum createAccumulator() { return new Accum(); }

    @Override
    public Accum addInput(Accum accum, Trip input) {
      accum.sum += input.getFare();
      accum.count++;
      return accum;
    }

    @Override
    public Accum mergeAccumulators(Iterable<Accum> accums) {
      Accum merged = createAccumulator();
      for (Accum accum : accums) {
        merged.sum += accum.sum;
        merged.count += accum.count;
      }
      return merged;
    }

    @Override
    public Double extractOutput(Accum accum) {
      return ((double) accum.sum) / accum.count;
    }
  }

  @VisibleForTesting
  static double roundAverageFare(double avg) {
    return Math.round(avg * 100) / 100d;
  }

  /**
   * Creates BigQuery's TableRow object for inserting into the table processed_trips.
   */
  private static class TripToProcessedTripTableRowConverter
          implements SerializableFunction<Trip, TableRow> {

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
    ValueProvider<String> getOutputDirectory();

    void setOutputDirectory(ValueProvider<String> outputDirectory);


    @Description("BigQuery dataset.")
    ValueProvider<String> getDataset();

    void setDataset(ValueProvider<String> dataset);


    @Description("BigQuery source table.")
    @Default.String("processed_trips")
    ValueProvider<String> getSourceTable();

    void setSourceTable(ValueProvider<String> sourceTable);


    @Description("A date from which the trips should be fetched from the source table.")
    @Default.String("2018-08-01")
    ValueProvider<String> getFromDate();

    void setFromDate(ValueProvider<String> date);


    @Description("Query limit.")
    @Default.Integer(2000000000)
    ValueProvider<Integer> getLimit();

    void setLimit(ValueProvider<Integer> limit);
  }

  private static String makeQuery(Options options) {
    return String.format(
            "SELECT t.unique_key, "
            + " DATETIME(TIMESTAMP_TRUNC(t.trip_start_timestamp, HOUR)) AS trip_start_hour, "
            + " t.pickup_latitude, "
            + " t.pickup_longitude, "
            + " t.pickup_community_area, "
            + " IF (h.countryRegionCode IS NULL, false, true) AS is_us_holiday, "
            + " t.fare "
            + " FROM `%1$s.%4$s` t "
            + " LEFT JOIN `%1$s.national_holidays` h "
            + "     ON TIMESTAMP_TRUNC(t.trip_start_timestamp, DAY) = TIMESTAMP_TRUNC(h.date, DAY) "
            + "     AND h.countryRegionCode = 'US' "
            + " WHERE trip_start_timestamp >= TIMESTAMP(\"%2$s 00:00:00+00\") "
            + " AND t.trip_start_timestamp IS NOT NULL "
            + " AND t.trip_end_timestamp IS NOT NULL "
            + " AND t.pickup_community_area IS NOT NULL "
            + " AND t.fare IS NOT NULL "
            + " AND t.trip_start_timestamp < t.trip_end_timestamp "
            + " AND t.unique_key NOT IN "
            + "     (SELECT unique_key FROM `%1$s.processed_trips`) "
            + " AND (SELECT "
            + "       ST_COVERS(b.boundaries, ST_GEOGPOINT(t.pickup_longitude, t.pickup_latitude)) "
            + "       FROM `%1$s.chicago_boundaries` b) = True "
            + " LIMIT %3$d",
    StringEscapeUtils.escapeSql(options.getDataset().get()),
    options.getFromDate().get(),
    options.getLimit().get(),
    StringEscapeUtils.escapeSql(options.getSourceTable().get()));
  }
}
