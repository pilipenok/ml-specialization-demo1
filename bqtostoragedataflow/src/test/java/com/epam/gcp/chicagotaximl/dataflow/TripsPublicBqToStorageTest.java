/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */
package com.epam.gcp.chicagotaximl.dataflow;


import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.AreaTripsDataToCsvConverter;
import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.TableRowToTripConverter;
import com.google.api.services.bigquery.model.TableSchema;
import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.beam.sdk.io.gcp.bigquery.SchemaAndRecord;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

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
        TestRecord record = new TestRecord();
        record.put("unique_key", "saaa");
        record.put("trip_start_hour", "2021-05-10T19:00");
        record.put("pickup_latitude", "41.11111111");
        record.put("pickup_longitude", "-87.454545");
        record.put("pickup_community_area", "77");
        record.put("is_us_holiday", "false");
        record.put("fare", "25.99");

        SchemaAndRecord schemaAndRecord = new SchemaAndRecord(record, new TableSchema());
        TableRowToTripConverter converter = new TableRowToTripConverter();

        Trip trip = converter.apply(schemaAndRecord);

        assertThat(trip.getUniqueKey()).isEqualTo("saaa");
        assertThat(trip.getTripStartHour()).isEqualTo(LocalDateTime.of(2021, 5, 10, 19, 0));
        assertThat(trip.getPickupLatitude()).isEqualTo(41.11111111d);
        assertThat(trip.getPickupLongitude()).isEqualTo(-87.454545d);
        assertThat(trip.getPickupArea()).isEqualTo(77);
        assertThat(trip.isUsHoliday()).isFalse();
        assertThat(trip.getFare()).isEqualTo(25.99f);
    }

    private static class TestRecord implements GenericRecord {

        private Map<String, Object> m = new HashMap<>();

        @Override
        public void put(String key, Object v) {
            m.put(key, v);
        }

        @Override
        public Object get(String key) {
            return m.get(key);
        }

        @Override
        public void put(int i, Object v) {
            // not in use
        }

        @Override
        public Object get(int i) { // not in use
            return null;
        }

        @Override
        public Schema getSchema() { // not in use
            return null;
        }
    }

    @Test
    public void testCreateAreaTripsData() {
        AreaTripsData areaTripsData = TripsPublicBqToStorage
                .createAreaTripsData(77, LocalDateTime.of(2021, 5, 8, 13, 0), 100, 30.12345, true);
        assertThat(areaTripsData.getAmPm()).isEqualTo("PM");
        assertThat(areaTripsData.getNumberOfTrips()).isEqualTo(100);
        assertThat(areaTripsData.getAverageFare()).isEqualTo(30.12345);
        assertThat(areaTripsData.getDayOfWeek()).isEqualTo(6);
        assertThat(areaTripsData.getMonth()).isEqualTo(5);
        assertThat(areaTripsData.getHourOfDay()).isEqualTo(1);
        assertThat(areaTripsData.getPickupCommunityArea()).isEqualTo(77);
    }
}
