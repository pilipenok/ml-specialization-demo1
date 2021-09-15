/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.cleanupfunction;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.QueryJobConfiguration.Builder;
import com.google.cloud.bigquery.TableResult;
import com.google.cloud.functions.BackgroundFunction;
import com.google.cloud.functions.Context;
import java.util.logging.Logger;
import org.apache.commons.lang.StringEscapeUtils;
import org.springframework.beans.factory.annotation.Autowired;

/**
 * Business logic for Cloud Function {@link CleanupFunctionStorageEvent}.
 * Update BigQuery table 'processed_trips' with current timestamp.
 */
public class CleanupFunction implements BackgroundFunction<GcsEvent> {

  private final String dataset;
  private final String filename;

  @Autowired
  private BigQuery bigquery;

  @Autowired
  private Logger logger;

  private static final String QUERY =
      "UPDATE `%1$s.processed_trips` "
          + "SET processed_timestamp = CURRENT_TIMESTAMP() "
          + "WHERE processed_timestamp IS NULL";

  public CleanupFunction(String dataset, String filename) {
    this.dataset = dataset;
    this.filename = filename;
  }

  @Override
  public void accept(GcsEvent event, Context context) throws Exception {
    if (!filename.equals(event.getName())) {
      logger.info(String.format("Ignored event on '%s'.", event.getName()));
      return;
    }
    Builder queryBuilder = QueryJobConfiguration.newBuilder(
        String.format(QUERY, StringEscapeUtils.escapeSql(dataset)));
    TableResult result = bigquery.query(queryBuilder.build());
    logger.info(String.format("%d rows marked as processed.", result.getTotalRows()));
  }
}
