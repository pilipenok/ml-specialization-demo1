/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.cleanupfunction.integration;

import static com.google.common.truth.Truth.assertThat;

import com.epam.gcp.chicagotaximl.cleanupfunction.CleanupFunction;
import com.epam.gcp.chicagotaximl.cleanupfunction.Config;
import com.epam.gcp.chicagotaximl.cleanupfunction.GcsEvent;
import com.google.cloud.Timestamp;
import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.FieldValueList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

/**
 * {@link CleanupFunction} with BigQuery integration test.
 */
@TestInstance(Lifecycle.PER_CLASS)
public class CleanupFunctionTest {

  private final AnnotationConfigApplicationContext ctx =
      new AnnotationConfigApplicationContext(TestConfig.class);

  private final BigQueryTesting bigQueryTesting = ctx.getBean(BigQueryTesting.class);
  private final CleanupFunction cleanupFunction = ctx.getBean(CleanupFunction.class);

  @BeforeAll
  void createBigQueryTable() throws Exception {
    bigQueryTesting.createTableMlDataset();
    bigQueryTesting.createTableProcessedTrips();
    bigQueryTesting.insertProcessedTrips(List.of(
        List.of("1", Timestamp.parseTimestamp("2020-12-01T10:15:30.000Z").toString()),
        List.of("2", ""),
        List.of("3", "")));
  }

  @AfterAll
  void destroy() {
    bigQueryTesting.close();
  }

  /**
   * Updated BigQuery table is expected.
   */
  @Test
  void testAccept() throws Exception {
    GcsEvent event = new GcsEvent();
    event.setName("trips/trips.csv");

    cleanupFunction.accept(event, null);

    List<FieldValueList> processedTripsResult = bigQueryTesting.query(
        String.format("SELECT * FROM %s.processed_trips ORDER BY unique_key",
            bigQueryTesting.getDataset()));
    assertThat(processedTripsResult).hasSize(3);
    FieldValueList row1 = processedTripsResult.get(0);
    FieldValueList row2 = processedTripsResult.get(1);
    FieldValueList row3 = processedTripsResult.get(2);

    assertThat(row1.get("unique_key").getStringValue()).isEqualTo("1");
    assertThat(row2.get("unique_key").getStringValue()).isEqualTo("2");
    assertThat(row3.get("unique_key").getStringValue()).isEqualTo("3");
    assertThat(row1.get("processed_timestamp").getTimestampValue())
        .isEqualTo(Timestamp.parseTimestamp("2020-12-01T10:15:30.000Z").getSeconds() * 1000000);
    assertThat(row2.get("processed_timestamp").getTimestampValue()).isNotNull();
    assertThat(row3.get("processed_timestamp").getTimestampValue()).isNotNull();
  }

  /**
   * Spring Framework test configuration.
   */
  @Configuration
  @ComponentScan("com.epam.gcp.chicagotaximl.cleanupfunction")
  static class TestConfig extends Config {

    private final BigQueryTesting bigQueryTesting = new BigQueryTesting();

    @Bean
    @Override
    public BigQuery createBigQuery() {
      return bigQueryTesting.getBigquery();
    }

    @Bean
    @Override
    public CleanupFunction createCleanupFunction() {
      return new CleanupFunction(bigQueryTesting.getDataset(), "trips/trips.csv");
    }

    @Bean
    public BigQueryTesting createBigQueryTesting() {
      return bigQueryTesting;
    }
  }
}
