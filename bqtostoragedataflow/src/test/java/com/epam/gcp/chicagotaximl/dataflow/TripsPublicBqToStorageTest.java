/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is,
 * and remains the property of EPAM Systems, Inc. and/or its suppliers and is protected by
 * international intellectual property law. Dissemination of this information or reproduction
 * of this material is strictly forbidden, unless prior written permission is obtained
 * from EPAM Systems, Inc
 */

package com.epam.gcp.chicagotaximl.dataflow;

import static com.google.common.truth.Truth.assertThat;

import com.epam.gcp.chicagotaximl.dataflow.TripsPublicBqToStorage.TableRowToCsv;
import com.google.api.services.bigquery.model.TableSchema;
import java.util.HashMap;
import java.util.Map;
import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.beam.sdk.io.gcp.bigquery.SchemaAndRecord;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link TripsPublicBqToStorage}
 */
class TripsPublicBqToStorageTest {

  @Test
  void testTableRowToCsv() {
    TestRecord record = new TestRecord();
    record.put("area", 7);
    record.put("year", 2021);
    record.put("quarter", 2);
    record.put("quarter_num", 3);
    record.put("quarter_cos", 4);
    record.put("quarter_sin", 5);
    record.put("month", 6);
    record.put("month_num", 7);
    record.put("month_cos", 8);
    record.put("month_sin", 9);
    record.put("day", 10);
    record.put("day_num", 11);
    record.put("day_cos", 12);
    record.put("day_sin", 13);
    record.put("hour", 14);
    record.put("hour_num", 15);
    record.put("hour_cos", 16);
    record.put("hour_sin", 17);
    record.put("day_period", "am");
    record.put("week", 19);
    record.put("week_num", 20);
    record.put("week_cos", 21);
    record.put("week_sin", 22);
    record.put("day_of_week", 23);
    record.put("day_of_week_num", 24);
    record.put("day_of_week_cos", 25);
    record.put("day_of_week_sin", 26);
    record.put("weekday_hour_num", 27);
    record.put("weekday_hour_cos", 28);
    record.put("weekday_hour_sin", 29);
    record.put("yearday_hour_num", 30);
    record.put("yearday_hour_cos", 31);
    record.put("yearday_hour_sin", 32);
    record.put("is_weekend", false);
    record.put("is_holiday", true);
    record.put("n_trips", 33);
    record.put("n_trips_num", 34);
    record.put("log_n_trips", 35);
    record.put("trips_bucket", 0);
    record.put("trips_bucket_num", 1);

    SchemaAndRecord schemaAndRecord = new SchemaAndRecord(record, new TableSchema());

    String result = new TableRowToCsv().apply(schemaAndRecord);

    assertThat(result).isEqualTo("7,2021,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,am,19,20,21,22,23,"
        + "24,25,26,27,28,29,30,31,32,false,true,33,34,35,0,1");
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
}
