/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */
package com.epam.gcp.chicagotaximl.dataflow;


import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.AreaTripsDataToCsvConverter;
import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.TableRowToTripConverter;
import com.google.api.services.bigquery.model.TableRow;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;

import static com.google.common.truth.Truth.assertThat;

/**
 * Unit tests for {@link TripsPublicBqToStorage}
 */
class TripsPublicBqToStorageTest {

    @Test
    public void testRoundAverageFare() {
        assertThat(TripsPublicBqToStorage.roundAverageFare(10.1234)).isEqualTo(10.12);
        assertThat(TripsPublicBqToStorage.roundAverageFare(1)).isEqualTo(1);
    }

    @Test
    public void testAreaTripsDataToCsvConverter() {
        AreaTripsData data = new AreaTripsData();
        data.setAmPm("AM");
        data.setAverageFare(20.14);
        data.setDayOfWeek(2);
        data.setNumberOfTrips(20l);
        data.setHourOfDay(3);
        data.setMonth(4);
        data.setPickupCommunityArea(77);
        data.setUsHoliday(true);
        AreaTripsDataToCsvConverter converter = new AreaTripsDataToCsvConverter();
        assertThat(converter.apply(data)).isEqualTo("77,2,true,4,3,AM,20.14,20");
    }

    @Test
    public void testMakeAreaHourGroupingKey() {
        Trip trip = new Trip("tripid");
        trip.setFare(20.18f);
        trip.setPickupArea(77);
        trip.setTripStartHour(LocalDateTime.of(2021, 5, 10, 19, 0));
        trip.setPickupLongitude(41.55555);
        trip.setPickupLongitude(-87.999999);
        trip.setUsHoliday(false);
        assertThat(TripsPublicBqToStorage.makeAreaHourGroupingKey(trip)).isEqualTo("77_2021-05-10T19:00");
    }

    @Test
    public void testTableRowToTripConverter() {
        TableRow row = new TableRow();
        row.set("unique_key", "saaa");
        row.set("trip_start_hour", "2021-05-10T19:00");
        row.set("pickup_latitude", "41.11111111");
        row.set("pickup_longitude", "-87.454545");
        row.set("pickup_community_area", "77");
        row.set("is_us_holiday", "false");
        row.set("fare", "25.99");

        TableRowToTripConverter converter = new TableRowToTripConverter();
        Trip trip = converter.apply(row);

        assertThat(trip.getUniqueKey()).isEqualTo("saaa");
        assertThat(trip.getTripStartHour()).isEqualTo(LocalDateTime.of(2021, 5, 10, 19, 0));
        assertThat(trip.getPickupLatitude()).isEqualTo(41.11111111d);
        assertThat(trip.getPickupLongitude()).isEqualTo(-87.454545d);
        assertThat(trip.getPickupArea()).isEqualTo(77);
        assertThat(trip.isUsHoliday()).isFalse();
        assertThat(trip.getFare()).isEqualTo(25.99f);
    }
}
