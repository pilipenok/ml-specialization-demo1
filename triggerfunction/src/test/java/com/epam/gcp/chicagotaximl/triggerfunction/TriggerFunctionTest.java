/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.dataflow.v1beta3.TemplatesServiceClient;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

/**
 * Unit test for {@link TriggerFunction}.
 */
public class TriggerFunctionTest {

  @Test
  void testRun_preprocessedStateFailed() throws Exception {
    AnnotationConfigApplicationContext ctx =
        new AnnotationConfigApplicationContext(TestConfig.class);
    TriggerFunction triggerFunction = ctx.getBean(TriggerFunction.class);
    Logger loggerMock = ctx.getBean(Logger.class);
    BigQueryDao bigQueryDaoMock = ctx.getBean(BigQueryDao.class);
    TemplatesServiceClient dataflowMock = ctx.getBean(TemplatesServiceClient.class);

    when(bigQueryDaoMock.verifyPreprocessedState()).thenReturn(false);

    triggerFunction.run();

    verify(bigQueryDaoMock).verifyPreprocessedState();
    verify(loggerMock).severe("Preprocessed trips state check failed.");

    verifyNoMoreInteractions(bigQueryDaoMock);
    verifyNoInteractions(dataflowMock);
  }

  @Test
  void testRun_lastModifiedTimeNotUpdated() throws Exception {
    AnnotationConfigApplicationContext ctx =
        new AnnotationConfigApplicationContext(TestConfig.class);
    TriggerFunction triggerFunction = ctx.getBean(TriggerFunction.class);
    BigQueryDao bigQueryDaoMock = ctx.getBean(BigQueryDao.class);
    TemplatesServiceClient dataflowMock = ctx.getBean(TemplatesServiceClient.class);

    when(bigQueryDaoMock.verifyPreprocessedState()).thenReturn(true);
    when(bigQueryDaoMock.verifyMlDatasetState()).thenReturn(true);
    when(bigQueryDaoMock.checkLastModifiedTime()).thenReturn(false);

    triggerFunction.run();

    verify(bigQueryDaoMock).verifyPreprocessedState();
    verify(bigQueryDaoMock).verifyMlDatasetState();
    verify(bigQueryDaoMock).checkLastModifiedTime();

    verifyNoMoreInteractions(bigQueryDaoMock);
    verifyNoInteractions(dataflowMock);
  }

  @Test
  void testRun_runDataflow() throws Exception {
    AnnotationConfigApplicationContext ctx =
        new AnnotationConfigApplicationContext(TestConfig.class);
    TriggerFunction triggerFunction = ctx.getBean(TriggerFunction.class);
    BigQueryDao bigQueryDaoMock = ctx.getBean(BigQueryDao.class);
    TemplatesServiceClient dataflowMock = ctx.getBean(TemplatesServiceClient.class);

    when(bigQueryDaoMock.verifyPreprocessedState()).thenReturn(true);
    when(bigQueryDaoMock.verifyMlDatasetState()).thenReturn(true);
    when(bigQueryDaoMock.checkLastModifiedTime()).thenReturn(true);

    triggerFunction.run();

    verify(bigQueryDaoMock).verifyPreprocessedState();
    verify(bigQueryDaoMock).verifyMlDatasetState();
    verify(bigQueryDaoMock).checkLastModifiedTime();
    verify(bigQueryDaoMock).prepareTripsForDataflow();
    verify(dataflowMock).createJobFromTemplate(any());

    verifyNoMoreInteractions(bigQueryDaoMock);
    verifyNoMoreInteractions(dataflowMock);
  }

  /**
   * Spring Framework test configuration.
   */
  @Configuration
  @ComponentScan("com.epam.gcp.chicagotaximl.triggerfunction")
  static class TestConfig extends Config {

    @Bean
    @Override
    public Logger createLogger() {
      return mock(Logger.class);
    }

    @Bean
    @Override
    public TemplatesServiceClient createTemplatesServiceClient()  {
      return mock(TemplatesServiceClient.class);
    }

    @Bean
    @Override
    public TriggerFunction createDataflowTrigger() {
      return new TriggerFunction(
                    "project",
                    "dataflow_job_name",
                    "gcs_path",
                    "temp_location",
                    "service_account",
                    "region");
    }

    @Bean
    @Override
    public BigQueryDao createBigQueryDao() {
      return mock(BigQueryDao.class);
    }
  }
}
