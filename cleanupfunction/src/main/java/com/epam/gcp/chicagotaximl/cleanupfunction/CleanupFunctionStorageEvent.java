/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.cleanupfunction;

import com.google.cloud.functions.BackgroundFunction;
import com.google.cloud.functions.Context;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

/**
 * Cloud Background Function wrapper for {@link CleanupFunction}.
 */
public class CleanupFunctionStorageEvent  implements BackgroundFunction<GcsEvent> {

  private CleanupFunction cleanupFunction;

  public CleanupFunctionStorageEvent() {
    AnnotationConfigApplicationContext ctx = new AnnotationConfigApplicationContext(Config.class);
    cleanupFunction = ctx.getBean(CleanupFunction.class);
  }

  @Override
  public void accept(GcsEvent event, Context context) throws Exception {
    cleanupFunction.accept(event, context);
  }
}
