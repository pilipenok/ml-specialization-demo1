/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */
package com.epam.gcp.chicagotaximl.dataflow;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.time.LocalDateTime;

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
    private Double fare;

    public Trip(String uniqueKey) {
        this.uniqueKey = uniqueKey;
    }
}
