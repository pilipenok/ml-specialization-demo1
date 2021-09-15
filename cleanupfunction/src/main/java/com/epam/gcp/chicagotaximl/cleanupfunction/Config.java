/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.cleanupfunction;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryOptions;
import java.util.logging.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

/**
 * Spring Framework configuration.
 */
@Configuration
@ComponentScan("com.epam.gcp.chicagotaximl.cleanupfunction")
public class Config {

  @Bean
  public Logger createLogger() {
    return Logger.getLogger(CleanupFunctionStorageEvent.class.getName());
  }

  @Bean
  public BigQuery createBigQuery() {
    return BigQueryOptions.getDefaultInstance().getService();
  }

  @Bean
  public CleanupFunction createCleanupFunction() {
    return new CleanupFunction(
        System.getenv("dataset"),
        System.getenv("filename"));
  }
}
