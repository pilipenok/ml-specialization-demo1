/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import com.google.dataflow.v1beta3.CreateJobFromTemplateRequest;
import com.google.dataflow.v1beta3.RuntimeEnvironment;
import com.google.dataflow.v1beta3.TemplatesServiceClient;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.springframework.beans.factory.annotation.Autowired;

/**
 * Checks if the public BigQuery table `taxi_trips` has been modified. If so, runs the
 * Dataflow job.
 */
public class TriggerFunction {

  @Autowired
  private TemplatesServiceClient dataflowClient;

  @Autowired
  private BigQueryDao bigQueryDao;

  @Autowired
  private Logger logger;

  private final String project;
  private final String dataflowJobName;
  private final String gcsPath;
  private final String tempLocation;
  private final String serviceAccount;
  private final String region;

  /**
   * The business logic implementation of Cloud Function 'TriggerFunction'.
   *
   * @param project Google Cloud project ID.
   * @param dataflowJobName Name of the Dataflow job.
   * @param gcsPath Dataflow template location.
   * @param tempLocation Dataflow temporary files location.
   * @param serviceAccount Dataflow service account.
   * @param region Dataflow region.
   */
  public TriggerFunction(String project,
      String dataflowJobName,
      String gcsPath,
      String tempLocation,
      String serviceAccount,
      String region) {
    this.project = project;
    this.dataflowJobName = dataflowJobName;
    this.gcsPath = gcsPath;
    this.tempLocation = tempLocation;
    this.serviceAccount = serviceAccount;
    this.region = region;
  }

  /**
   * Execute function.
   */
  public void run() {
    try {
      if (!bigQueryDao.verifyPreprocessedState()) {
        logger.severe("Preprocessed trips state check failed.");
        return;
      }

      if (!bigQueryDao.verifyMlDatasetState()) {
        logger.severe("Dataset table state check failed.");
        return;
      }

      logger.info("Checking last_modified_time.");
      if (!bigQueryDao.checkLastModifiedTime()) {
        return;
      }

      logger.info("Preparing new trips.");
      bigQueryDao.prepareTripsForDataflow();

      logger.info("Running Dataflow job.");
      runDataflow();
    } catch (Exception e) {
      logger.log(Level.SEVERE, "", e);
    }
  }

  /**
   * Starts the Dataflow job.
   */
  private void runDataflow() {
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
