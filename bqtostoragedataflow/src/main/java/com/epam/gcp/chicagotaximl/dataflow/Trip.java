/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.dataflow;

import java.io.Serializable;
import java.time.LocalDateTime;
import lombok.Getter;
import lombok.Setter;

/**
 * Contains data about trip.
 */
@Setter
@Getter
public class Trip implements Serializable {

  private final String uniqueKey;

  private LocalDateTime tripStartHour;
  private Double pickupLatitude;
  private Double pickupLongitude;
  private Integer pickupArea;
  private boolean isUsHoliday;
  private Float fare;

  public Trip(String uniqueKey) {
    this.uniqueKey = uniqueKey;
  }

  @Override
  public int hashCode() {
    return uniqueKey.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Trip)) {
      return false;
    }
    return ((Trip) o).getUniqueKey().equals(uniqueKey);
  }
}
