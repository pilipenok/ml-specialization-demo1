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
import com.google.cloud.functions.HttpFunction;
import com.google.cloud.functions.HttpRequest;
import com.google.cloud.functions.HttpResponse;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import lombok.AllArgsConstructor;

/**
 * Cloud Function with http trigger. Runs the Dataflow job if the source table has been modified.
 */
@AllArgsConstructor
public class DataflowTriggerHttpFunction implements HttpFunction {

  private static final Logger LOGGER = Logger.getLogger(DataflowTrigger.class.getName());
  private static final Gson GSON = new Gson();
  private static final String CHECK_LAST_MODIFIED_PARAM = "check-last-modified-time";

  private final DataflowClientWrapper dataflowClient;
  private final BigQuery bigquery;

  /**
   * Initializes client libraries.
   */
  public DataflowTriggerHttpFunction() throws IOException {
    this.dataflowClient = new DataflowClientWrapper();
    this.bigquery = BigQueryOptions.getDefaultInstance().getService();
  }

  @Override
  public void service(HttpRequest request, HttpResponse response) {
    try {
      JsonObject body = GSON.fromJson(request.getReader(), JsonObject.class);

      boolean checkLastModifiedTime = body.has(CHECK_LAST_MODIFIED_PARAM)
          ? body.get(CHECK_LAST_MODIFIED_PARAM).getAsBoolean()
          : true;
      boolean runDataflow = true;

      DataflowTrigger dataflowTrigger = new DataflowTrigger(
          dataflowClient,
          bigquery,
          body.get("project").getAsString(),
          body.get("dataflow-job-name").getAsString(),
          body.get("gcs-path").getAsString(),
          body.get("temp-location").getAsString(),
          body.get("service-account").getAsString(),
          body.get("region").getAsString());

      if (checkLastModifiedTime) {
        LOGGER.info("Checking last_modified_time");
        runDataflow = dataflowTrigger.checkLastModifiedTime();
      }
      if (runDataflow) {
        LOGGER.info("Running Dataflow job");
        dataflowTrigger.runDataflow();
      }
    } catch (Exception e) {
      LOGGER.log(Level.SEVERE, "", e);
    }
  }
}
