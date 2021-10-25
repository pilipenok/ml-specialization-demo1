/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction.integration;

import static com.google.common.truth.Truth.assertThat;

import com.epam.gcp.chicagotaximl.triggerfunction.BigQueryDao;
import com.epam.gcp.chicagotaximl.triggerfunction.Config;
import com.google.cloud.Timestamp;
import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.FieldValueList;
import java.time.LocalDate;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;

/**
 * {@link BigQueryDao} with BigQuery integration test.
 */
@TestInstance(Lifecycle.PER_CLASS)
public class BigQueryDaoTest {

  private final AnnotationConfigApplicationContext ctx =
      new AnnotationConfigApplicationContext(TestConfig.class);

  private final BigQueryDao bigQueryDao = ctx.getBean(BigQueryDao.class);
  private final BigQueryTesting bigQueryTesting = ctx.getBean(BigQueryTesting.class);

  @BeforeAll
  void createBigQueryTable() throws Exception {
    bigQueryTesting.createTableTaxiTrips();
    bigQueryTesting.createTableProcessedTrips();
    bigQueryTesting.createTablePublicDatasetTables();
    bigQueryTesting.createTableSourceModificationTimestamps();
    bigQueryTesting.createTableChicagoBoundaries();
    bigQueryTesting.fillTableChicagoBoundaries();
  }

  @AfterAll
  void destroy() {
    bigQueryTesting.close();
  }

  @AfterEach
  void truncate() throws Exception {
    bigQueryTesting.truncateTaxiTrips();
    bigQueryTesting.truncateProcessedTrips();
    bigQueryTesting.truncatePublicDatasetTables();
    bigQueryTesting.truncateSourceModificationTimestamps();
  }

  /**
   * First run. Table 'source_modification_timestamps' is empty.
   * Excepted True (modified).
   */
  @Test
  void testCheckLastModifiedTime_firstRun() throws Exception {
    bigQueryTesting.insertPublicDatasetTables(List.of(
        List.of("some_table", String.valueOf(
            Timestamp.parseTimestamp("2021-10-01T10:15:30.000Z").getSeconds() * 1000)),
        List.of("taxi_trips", String.valueOf(
            Timestamp.parseTimestamp("2021-08-01T10:15:30.000Z").getSeconds() * 1000))));

    assertThat(bigQueryDao.checkLastModifiedTime()).isTrue();
  }

  /**
   * Saved sourceModificationTimestamp is in the past, expected True (modified).
   */
  @Test
  void testCheckLastModifiedTime_true() throws Exception {
    bigQueryTesting.insertPublicDatasetTables(List.of(
        List.of("some_table", String.valueOf(
            Timestamp.parseTimestamp("2021-10-01T10:15:30.000Z").getSeconds() * 1000)),
        List.of("taxi_trips", String.valueOf(
            Timestamp.parseTimestamp("2021-08-01T10:15:30.000Z").getSeconds() * 1000))));

    bigQueryTesting.insertSourceModificationTimestamps(
        List.of(List.of(Timestamp.parseTimestamp("2021-07-01T10:15:30.000Z").toString())));
    List<FieldValueList> res = bigQueryTesting.query(
        String.format("SELECT * FROM %s.%s", bigQueryTesting.getDataset(),
            BigQueryTesting.SOURCE_MODIFICATION_TABLE));

    assertThat(bigQueryDao.checkLastModifiedTime()).isTrue();
  }

  /**
   * Saved sourceModificationTimestamp is equal to the timestamp in public_dataset_tables,
   * expected False (not modified).
   */
  @Test
  void testCheckLastModifiedTime_false() throws Exception {
    bigQueryTesting.insertPublicDatasetTables(List.of(
        List.of("some_table", String.valueOf(
            Timestamp.parseTimestamp("2021-10-01T10:15:30.000Z").getSeconds() * 1000)),
        List.of("taxi_trips", String.valueOf(
            Timestamp.parseTimestamp("2021-08-01T10:15:30.000Z").getSeconds() * 1000))));

    bigQueryTesting.insertSourceModificationTimestamps(
        List.of(List.of(Timestamp.parseTimestamp("2021-08-01T10:15:30.000Z").toString())));
    List<FieldValueList> res = bigQueryTesting.query(
        String.format("SELECT * FROM %s.%s", bigQueryTesting.getDataset(),
            BigQueryTesting.SOURCE_MODIFICATION_TABLE));

    assertThat(bigQueryDao.checkLastModifiedTime()).isFalse();
  }

  /**
   * First run. Table processed_trips is empty. Expected True (valid state).
   */
  @Test
  void testVerifyPreprocessedState_firstRun() throws Exception {
    assertThat(bigQueryDao.verifyPreprocessedState()).isTrue();
  }

  /**
   * All records in the table 'processed_trips' have a timestamp. Expected True (valid state).
   */
  @Test
  void testVerifyPreprocessedState_pass() throws Exception {
    bigQueryTesting.insertProcessedTrips(List.of(
        List.of("1", Timestamp.parseTimestamp("2021-09-01T10:15:30.000Z").toString())));

    assertThat(bigQueryDao.verifyPreprocessedState()).isTrue();
  }

  /**
   * Table 'processed_trips' has a record with empty timestamp. Expected False (invalid state).
   */
  @Test
  void testVerifyPreprocessedState_fail() throws Exception {
    bigQueryTesting.insertProcessedTrips(List.of(
        List.of("1", Timestamp.parseTimestamp("2021-09-01T10:15:30.000Z").toString()),
        List.of("2", "")));

    assertThat(bigQueryDao.verifyPreprocessedState()).isFalse();
  }

  @Test
  void testNewModificationTimestampQuery() throws Exception {
    bigQueryTesting.insertPublicDatasetTables(List.of(
        List.of("some_table", String.valueOf(
            Timestamp.parseTimestamp("2021-10-01T10:15:30.000Z").getSeconds() * 1000)),
        List.of("taxi_trips", String.valueOf(
            Timestamp.parseTimestamp("2021-08-10T10:15:30.000Z").getSeconds() * 1000))));

    String query = String.format(BigQueryDao.NEW_MODIFICATION_TIMESTAMP_QUERY,
        bigQueryTesting.getDataset());
    bigQueryTesting.query(query);

    List<FieldValueList> modificationTimestamps = bigQueryTesting.query(
        String.format("SELECT * FROM `%s.%s`", bigQueryTesting.getDataset(),
            BigQueryTesting.SOURCE_MODIFICATION_TABLE));
    assertThat(modificationTimestamps).hasSize(1);
    assertThat(modificationTimestamps.get(0).get("modification_timestamp").getTimestampValue())
        .isEqualTo(Timestamp.parseTimestamp("2021-08-10T10:15:30.000Z").getSeconds() * 1000000);
  }

  /**
   * First run. Tables 'processed_trips' and 'source_modification_timestamps' are empty.
   */
  @Test
  void testNewTripsQuery_firstRun() throws Exception {
    bigQueryTesting.insertTaxiTrips(List.of(
        List.of("3", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "41.968069", "-87.721559063", "41.968069", "-87.721559063"),
        List.of("4", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "41.968069", "-87.721559063", "41.968069", "-87.721559063"))
    );

    String query = String.format(BigQueryDao.NEW_TRIPS_QUERY,
        bigQueryTesting.getDataset(), "taxi_trips", "2020-04-01");
    bigQueryTesting.query(query);

    // Expected new records in table 'processed_trips'

    List<FieldValueList> processedTrips = bigQueryTesting.query(
        String.format("SELECT * FROM `%s.%s` ORDER BY unique_key", bigQueryTesting.getDataset(),
            BigQueryTesting.PROCESSED_TRIPS_TABLE));
    assertThat(processedTrips).hasSize(2);
    assertThat(processedTrips.get(0).get("unique_key").getStringValue()).isEqualTo("3");
    assertThat(processedTrips.get(1).get("unique_key").getStringValue()).isEqualTo("4");
  }

  @Test
  void testNewTripsQuery() throws Exception {
    // Trips have different IDs only
    bigQueryTesting.insertTaxiTrips(List.of(
        List.of("1", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "41.968069", "-87.721559063", "41.968069", "-87.721559063"),
        List.of("2", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "41.968069", "-87.721559063", "41.968069", "-87.721559063"),
        List.of("3", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "41.968069", "-87.721559063", "41.968069", "-87.721559063"),
        List.of("4", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "41.968069", "-87.721559063", "41.968069", "-87.721559063"))
    );

    bigQueryTesting.insertProcessedTrips(List.of(
        List.of("1", Timestamp.parseTimestamp("2021-08-09T11:00:00.000Z").toString()),
        List.of("2", Timestamp.parseTimestamp("2021-08-09T11:00:00.000Z").toString())));

    String query = String.format(BigQueryDao.NEW_TRIPS_QUERY,
        bigQueryTesting.getDataset(), "taxi_trips", "2020-04-01");
    bigQueryTesting.query(query);

    // Table 'processed_trips' has already had 2 records, expecting 2 more records to appear

    List<FieldValueList> processedTrips = bigQueryTesting.query(
        String.format("SELECT * FROM `%s.%s` ORDER BY unique_key", bigQueryTesting.getDataset(),
            BigQueryTesting.PROCESSED_TRIPS_TABLE));
    assertThat(processedTrips).hasSize(4);
    FieldValueList processedTrip1 = processedTrips.get(0);
    FieldValueList processedTrip2 = processedTrips.get(1);
    FieldValueList processedTrip3 = processedTrips.get(2);
    FieldValueList processedTrip4 = processedTrips.get(3);

    assertThat(processedTrip1.get("unique_key").getStringValue()).isEqualTo("1");
    assertThat(processedTrip1.get("processed_timestamp").getTimestampValue())
        .isEqualTo(Timestamp.parseTimestamp("2021-08-09T11:00:00.000Z").getSeconds() * 1000000);
    assertThat(processedTrip2.get("unique_key").getStringValue()).isEqualTo("2");
    assertThat(processedTrip2.get("processed_timestamp").getTimestampValue())
        .isEqualTo(Timestamp.parseTimestamp("2021-08-09T11:00:00.000Z").getSeconds() * 1000000);
    assertThat(processedTrip3.get("unique_key").getStringValue()).isEqualTo("3");
    assertThat(processedTrip3.get("processed_timestamp").isNull()).isTrue();
    assertThat(processedTrip4.get("unique_key").getStringValue()).isEqualTo("4");
    assertThat(processedTrip4.get("processed_timestamp").isNull()).isTrue();
  }

  @Test
  void testPrepareTripsForDataflow_invalidGeoPoints() throws Exception {
    bigQueryTesting.insertTaxiTrips(List.of(
        List.of("3", "taxi_id",
            Timestamp.parseTimestamp("2021-05-01T20:10:05.000Z").toString(),
            Timestamp.parseTimestamp("2021-05-01T20:40:05.000Z").toString(),
            "1200", "15", // length
            "7", "8", // areas
            "25.50", "30.50", "card", // payment
            "1", "1", "41.968069", "-87.721559063"))
    );

    String query = String.format(BigQueryDao.NEW_TRIPS_QUERY,
        bigQueryTesting.getDataset(), "taxi_trips", "2020-04-01");
    bigQueryTesting.query(query);

    // Expected no new records added in table 'processed_trips'

    List<FieldValueList> processedTrips = bigQueryTesting.query(
        String.format("SELECT * FROM `%s.%s` ORDER BY unique_key", bigQueryTesting.getDataset(),
            BigQueryTesting.PROCESSED_TRIPS_TABLE));
    assertThat(processedTrips).isEmpty();
  }

  /**
   * Spring Framework test configuration.
   */
  private static class TestConfig extends Config {

    private final BigQueryTesting bigQueryTesting = new BigQueryTesting();

    @Bean
    @Override
    public BigQuery createBigQuery() {
      return bigQueryTesting.getBigquery();
    }

    @Bean
    @Override
    public BigQueryDao createBigQueryDao() {
      return new BigQueryDao(
          bigQueryTesting.getDataset(),
          BigQueryTesting.TAXI_TRIPS_TABLE,
          LocalDate.parse("2020-04-01"));
    }

    @Bean
    public BigQueryTesting createBigQueryTesting() {
      return bigQueryTesting;
    }
  }
}
