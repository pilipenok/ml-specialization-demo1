/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.cleanupfunction;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.TableResult;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

/**
 * Unit test for {@link CleanupFunction}.
 */
@TestInstance(Lifecycle.PER_CLASS)
public class CleanupFunctionTest {

  private final AnnotationConfigApplicationContext ctx =
      new AnnotationConfigApplicationContext(TestConfig.class);

  private final CleanupFunction cleanupFunction = ctx.getBean(CleanupFunction.class);
  private final Logger loggerMock = ctx.getBean(Logger.class);
  private final BigQuery bigQueryMock = ctx.getBean(BigQuery.class);

  @Test
  void testAccept_ignoreWrongFilename() throws Exception {
    GcsEvent event = new GcsEvent();
    event.setName("AAA");
    cleanupFunction.accept(event, null);

    verify(loggerMock).info("Ignored event on 'AAA'.");
    verifyNoInteractions(bigQueryMock);
  }

  @Test
  void testAccept_success() throws Exception {
    TableResult result = mock(TableResult.class);
    when(result.getTotalRows()).thenReturn((long) 25);
    when(bigQueryMock.query(any())).thenReturn(result);

    GcsEvent event = new GcsEvent();
    event.setName("trips/trips.csv");
    cleanupFunction.accept(event, null);

    verify(bigQueryMock).query(any());
    verify(loggerMock).info("25 rows marked as processed.");
  }

  /**
   * Spring Framework test configuration.
   */
  @Configuration
  @ComponentScan("com.epam.gcp.chicagotaximl.cleanupfunction")
  static class TestConfig extends Config {

    @Bean
    @Override
    public Logger createLogger() {
      return  mock(Logger.class);
    }

    @Bean
    @Override
    public BigQuery createBigQuery() {
      return mock(BigQuery.class);
    }

    @Bean
    @Override
    public CleanupFunction createCleanupFunction() {
      return new CleanupFunction("mydataset", "trips/trips.csv");
    }
  }
}
