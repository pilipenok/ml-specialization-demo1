/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.triggerfunction;

import com.google.cloud.functions.BackgroundFunction;
import com.google.cloud.functions.Context;
import com.google.events.cloud.pubsub.v1.Message;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

/**
 * Cloud Function with a Pub/Sub trigger. Runs the Dataflow job if the source table
 * has been modified.
 */
public class TriggerFunctionPubSubEvent implements BackgroundFunction<Message> {

  private final TriggerFunction dataflowTrigger;

  public TriggerFunctionPubSubEvent() {
    AnnotationConfigApplicationContext ctx = new AnnotationConfigApplicationContext(Config.class);
    dataflowTrigger = ctx.getBean(TriggerFunction.class);
  }

  @Override
  public void accept(Message message, Context context) {
    dataflowTrigger.run();
  }
}
