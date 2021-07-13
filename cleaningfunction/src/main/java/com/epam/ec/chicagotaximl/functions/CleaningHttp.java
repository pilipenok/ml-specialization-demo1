/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */

package com.epam.ec.chicagotaximl.functions;

import com.google.cloud.functions.HttpFunction;
import com.google.cloud.functions.HttpRequest;
import com.google.cloud.functions.HttpResponse;

/**
 * Cloud Function with http trigger. Moves the processed .csv files to another (archive) bucket of Cloud Storage.
 */
public class CleaningHttp implements HttpFunction {

  @Override
  public void service(HttpRequest httpRequest, HttpResponse httpResponse) {
    StorageCleaning storageCleaning = new StorageCleaning(
            "chicago-taxi-ml-demo-1",
            "trips/trips-",
            "chicago-taxi-ml-demo-1-archive",
            "trips");
    storageCleaning.moveFilesToArchive();
  }
}
