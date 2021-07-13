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
 * Contains trips data for a particular community area in a particular hour.
 */
@Setter
@Getter
public class AreaTripsData implements Serializable {

    private Integer pickupCommunityArea;
    private Long numberOfTrips;
    private Double averageFare;
    private Integer month;
    private Integer dayOfWeek;
    private Integer hourOfDay;
    private boolean isUsHoliday;
    private String amPm;
}
