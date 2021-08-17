/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import com.google.dataflow.v1beta3.CreateJobFromTemplateRequest;
import com.google.dataflow.v1beta3.Job;
import com.google.dataflow.v1beta3.TemplatesServiceClient;
import java.io.IOException;

/**
 * Wrapper for TemplatesServiceClient. The only purpose of the wrapper is to be able to mock
 * TemplatesServiceClient in unit test.
 */
public class DataflowClientWrapper {

  private final TemplatesServiceClient templatesServiceClient;

  /**
   * Initiates TemplatesServiceClient.
   */
  public DataflowClientWrapper() throws IOException {
    templatesServiceClient = TemplatesServiceClient.create();
  }

  /**
   * Calls createJobFromTemplate of TemplatesServiceClient.
   */
  public Job createJobFromTemplate(CreateJobFromTemplateRequest request) {
    return templatesServiceClient.createJobFromTemplate(request);
  }
}
