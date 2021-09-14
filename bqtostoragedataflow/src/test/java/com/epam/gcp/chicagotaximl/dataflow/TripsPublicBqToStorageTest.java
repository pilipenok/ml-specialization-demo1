/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.dataflow;

import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.KvToCsvConverter;
import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.TableRowToTripConverter;
import com.google.api.services.bigquery.model.TableSchema;
import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.beam.sdk.io.gcp.bigquery.SchemaAndRecord;
import org.apache.beam.sdk.transforms.join.CoGbkResult;
import org.apache.beam.sdk.transforms.join.CoGbkResultSchema;
import org.apache.beam.sdk.transforms.join.RawUnionValue;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.TupleTag;
import org.apache.beam.sdk.values.TupleTagList;
import org.junit.jupiter.api.Test;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.google.common.truth.Truth.assertThat;

/**
 * Unit tests for {@link TripsPublicBqToStorage}
 */
class TripsPublicBqToStorageTest {

  @Test
  void testRoundAverageFare() {
    assertThat(TripsPublicBqToStorage.roundAverageFare(10.1234)).isEqualTo(10.12);
    assertThat(TripsPublicBqToStorage.roundAverageFare(1)).isEqualTo(1);
  }

  @Test
  void testMakeAreaHourGroupingKey() {
    Trip trip = createTrip("tripid", 20.18f, 77,
        LocalDateTime.of(2021, 5, 10, 19, 0), 41.55555, -87.999999, false);
    assertThat(TripsPublicBqToStorage.makeAreaHourGroupingKey(trip))
        .isEqualTo("77_2021-05-10T19:00");
  }

  @Test
  void testTableRowToTripConverter() {
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
  void testKvToCsvConversion() {
    final TupleTag<Trip> tripsTag = new TupleTag<>();
    final TupleTag<Long> countsTag = new TupleTag<>();
    final TupleTag<Double> averagesTag = new TupleTag<>();

    TupleTagList tags = TupleTagList.of(tripsTag).and(countsTag).and(averagesTag);
    CoGbkResultSchema schema = new CoGbkResultSchema(tags);

    // Grouped values: area 77, 2021-12-01 1pm-2pm, average fare 18.1
    RawUnionValue trip1Value1 = new RawUnionValue(0,
      createTrip("1", 23.5f, 77, LocalDateTime.of(2021, 12, 1, 13, 5, 0), 41.111, -87.555, true));
    RawUnionValue trip2Value1 = new RawUnionValue(0,
      createTrip("2", 12.7f, 77, LocalDateTime.of(2021, 12, 1, 13, 10, 0), 41.222, -87.666, true));
    RawUnionValue countsValue1 = new RawUnionValue(1, 2l);
    RawUnionValue averageFareValue1 = new RawUnionValue(2, 18.1d);

    CoGbkResult result = new CoGbkResult(schema, List.of(
                trip1Value1, trip2Value1, countsValue1, averageFareValue1));

    KV<String, CoGbkResult> kv = KV.of("77_2021-12-01T13:00:00", result);

    assertThat(new KvToCsvConverter(tripsTag, countsTag, averagesTag).apply(kv))
                .isEqualTo("77,3,true,12,1,PM,18.1,2");
  }

  private static Trip createTrip(String id, float fare, int pickupArea,
      LocalDateTime tripStartHour, double pickupLatitude,
      double pickupLongitude, boolean isUsHoliday) {
    Trip trip = new Trip(id);
    trip.setFare(fare);
    trip.setPickupArea(pickupArea);
    trip.setTripStartHour(tripStartHour);
    trip.setPickupLatitude(pickupLatitude);
    trip.setPickupLongitude(pickupLongitude);
    trip.setUsHoliday(isUsHoliday);
    return trip;
  }
}
