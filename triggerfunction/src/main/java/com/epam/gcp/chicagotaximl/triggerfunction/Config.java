/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryOptions;
import com.google.dataflow.v1beta3.TemplatesServiceClient;
import java.io.IOException;
import java.time.LocalDate;
import java.util.logging.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

/**
 * Spring Framework configuration.
 * Cloud Functions only finds injection binding if it's configured in a provider method.
 */
@Configuration
@ComponentScan("com.epam.gcp.chicagotaximl.triggerfunction")
public class Config {

  @Bean
  public Logger createLogger() {
    return Logger.getLogger(TriggerFunctionPubSubEvent.class.getName());
  }

  @Bean
  public BigQuery createBigQuery() {
    return BigQueryOptions.getDefaultInstance().getService();
  }

  @Bean
  public TemplatesServiceClient createTemplatesServiceClient() throws IOException {
    return TemplatesServiceClient.create();
  }

  /**
   * DI provider for {@link TriggerFunction}.
   */
  @Bean
  public TriggerFunction createDataflowTrigger() {
    return new TriggerFunction(
        System.getenv("project"),
        System.getenv("dataflow_job_name"),
        System.getenv("gcs_path"),
        System.getenv("temp_location"),
        System.getenv("service_account"),
        System.getenv("region"));
  }

  /**
   * DI provider for {@link BigQueryDao}.
   */
  @Bean
  public BigQueryDao createBigQueryDao() {
    return new BigQueryDao(
        System.getenv("dataset"),
        System.getenv("source_table"),
        LocalDate.parse(System.getenv("start_date")));
  }
}
