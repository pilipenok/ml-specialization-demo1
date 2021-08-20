/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.QueryJobConfiguration.Builder;
import com.google.cloud.bigquery.TableResult;
import com.google.dataflow.v1beta3.CreateJobFromTemplateRequest;
import com.google.dataflow.v1beta3.RuntimeEnvironment;
import java.util.HashMap;
import lombok.AllArgsConstructor;
import org.apache.commons.lang.StringEscapeUtils;

/**
 * Checks if the public BigQuery table `taxi_trips` has been modified. If so, runs the
 * Dataflow job.
 */
@AllArgsConstructor
public class DataflowTrigger {

  private final DataflowClientWrapper dataflowClient;
  private final BigQuery bigquery;
  private final String project;
  private final String dataflowJobName;
  private final String gcsPath;
  private final String tempLocation;
  private final String serviceAccount;
  private final String region;
  private final String dataset;

  // public_dataset_tables is a view for table bigquery-public-data.chicago_taxi_trips.__TABLES__,
  // which is a system table and contains information about tables in a dataset.
  // The query returns 1 row if last modification time of the table
  // bigquery-public-data.chicago_taxi_trips.taxi_trips is greater than the maximum value of
  // the taxi_id.processed_timestamp (minus 7 min, because the Dataflow running time might be up to
  // 7 minutes) or the table taxi_id is empty.
  private static final String QUERY =
      "SELECT last_modified_time "
          + "FROM `%1$s.public_dataset_tables` "
          + "WHERE table_id = 'taxi_trips' "
          + "AND (TIMESTAMP_MILLIS(last_modified_time) > "
          + "         (SELECT TIMESTAMP_SUB(MAX(processed_timestamp), INTERVAL 7 MINUTE) "
          + "           FROM `%1$s.processed_trips`) "
          + "      OR (SELECT processed_timestamp FROM `%1$s.processed_trips` LIMIT 1) IS NULL "
          + "     )";

  /**
   * Returns True if BigQuery table 'bigquery-public-data.chicago_taxi_trips.taxi_trips'
   * has been modified since previous Dataflow run.
   */
  public boolean checkLastModifiedTime() throws InterruptedException {
    Builder queryBuilder = QueryJobConfiguration.newBuilder(
        String.format(QUERY, StringEscapeUtils.escapeSql(dataset)));
    TableResult results = bigquery.query(queryBuilder.build());
    return results.getTotalRows() > 0;
  }

  /**
   * Starts the Dataflow job.
   */
  public void runDataflow() {
    CreateJobFromTemplateRequest request =
        CreateJobFromTemplateRequest.newBuilder()
            .setProjectId(project)
            .setJobName(dataflowJobName)
            .setGcsPath(gcsPath)
            .putAllParameters(new HashMap<String, String>())
            .setEnvironment(
                RuntimeEnvironment.newBuilder()
                    .setServiceAccountEmail(serviceAccount)
                    .setTempLocation(tempLocation)
                    .build())
            .setLocation(region)
            .build();
    dataflowClient.createJobFromTemplate(request);
  }
}
