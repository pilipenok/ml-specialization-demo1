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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import com.google.api.gax.paging.Page;
import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.TableResult;
import com.google.cloud.functions.HttpRequest;
import com.google.cloud.functions.HttpResponse;
import java.io.BufferedReader;
import java.io.StringReader;
import org.junit.jupiter.api.Test;

/**
 * Unit test for {@link DataflowTriggerHttpFunction}.
 * In this test we avoid creating the Dataflow job, the reason is Mockito doesn't mock
 * the Dataflow client library properly and fails with the NullPointerException.
 */
public class DataflowTriggerHttpFunctionTest {

  private static final StringBuilder requestBuilder = new StringBuilder("{"
      + " 'project': 'myproject',"
      + " 'dataflow-job-name': 'my-dataflow-job-name',"
      + " 'gcs-path': 'my-gcs-path',"
      + " 'temp-location': 'my-temp-location',"
      + " 'service-account': 'my-service-account',"
      + " 'region': 'my-region'"
      + "}");
  private static final String REQUEST_NO_CHECK_TIME = requestBuilder.toString();
  private static final String REQUEST_CHECK_TIME_FALSE = requestBuilder.insert(
      requestBuilder.length() - 1, ", 'check-last-modified-time': false").toString();
  private static final String REQUEST_CHECK_TIME_TRUE = requestBuilder.insert(
      requestBuilder.length() - 1, ", 'check-last-modified-time': true").toString();

  private static final TableResult BQ_EMPTY_RES = new TableResult(null, 0, mock(Page.class));
  private static final TableResult BQ_NON_EMPTY_RES = new TableResult(null, 1, mock(Page.class));

  private final DataflowClientWrapper dataflowClientWrapperMock = mock(DataflowClientWrapper.class);
  private final BigQuery bigqueryMock = mock(BigQuery.class);
  private final HttpResponse httpResponseMock = mock(HttpResponse.class);
  private final HttpRequest httpRequestMock = mock(HttpRequest.class);

  /**
   * No flag check-last-modified-time, checkLastModified returns False.
   * Expected BigQuery request and Dataflow not started.
   */
  @Test
  void testService_noCheckModifiedTimeFlag() throws Exception {
    when(httpRequestMock.getReader())
        .thenReturn(new BufferedReader(new StringReader(REQUEST_NO_CHECK_TIME)));
    when(bigqueryMock.query(any())).thenReturn(BQ_EMPTY_RES);

    DataflowTriggerHttpFunction function =
        new DataflowTriggerHttpFunction(dataflowClientWrapperMock, bigqueryMock);
    function.service(httpRequestMock, httpResponseMock);

    verify(bigqueryMock, times(1)).query(any());
    verifyNoInteractions(dataflowClientWrapperMock);
  }

  /**
   * Flag check-last-modified-time is True, checkLastModified returns False.
   * Expected BigQuery request and Dataflow not started.
   */
  @Test
  void testService_checkModifiedFlagTrue() throws Exception {
    when(httpRequestMock.getReader())
        .thenReturn(new BufferedReader(new StringReader(REQUEST_CHECK_TIME_TRUE)));
    when(bigqueryMock.query(any())).thenReturn(BQ_EMPTY_RES);

    DataflowTriggerHttpFunction function =
        new DataflowTriggerHttpFunction(dataflowClientWrapperMock, bigqueryMock);
    function.service(httpRequestMock, httpResponseMock);

    verify(bigqueryMock, times(1)).query(any());
    verifyNoInteractions(dataflowClientWrapperMock);
  }

  /**
   * Flag check-last-modified-time is True, checkLastModified returns True.
   * Expected BigQuery request and Dataflow runs.
   */
  @Test
  void testService_checkModifiedFlagTrueResPositive() throws Exception {
    when(httpRequestMock.getReader())
        .thenReturn(new BufferedReader(new StringReader(REQUEST_CHECK_TIME_TRUE)));
    when(bigqueryMock.query(any())).thenReturn(BQ_NON_EMPTY_RES);

    DataflowTriggerHttpFunction function =
        new DataflowTriggerHttpFunction(dataflowClientWrapperMock, bigqueryMock);
    function.service(httpRequestMock, httpResponseMock);

    verify(bigqueryMock, times(1)).query(any());
    verify(dataflowClientWrapperMock, times(1)).createJobFromTemplate(any());
  }

  /**
   * Flag check-last-modified-time is False.
   * Expected no BigQuery request and Dataflow runs.
   */
  @Test
  void testService_checkModifiedFlagFalse() throws Exception {
    when(httpRequestMock.getReader())
        .thenReturn(new BufferedReader(new StringReader(REQUEST_CHECK_TIME_FALSE)));

    DataflowTriggerHttpFunction function =
        new DataflowTriggerHttpFunction(dataflowClientWrapperMock, bigqueryMock);
    function.service(httpRequestMock, httpResponseMock);

    verifyNoInteractions(bigqueryMock);
    verify(dataflowClientWrapperMock, times(1)).createJobFromTemplate(any());
  }
}
