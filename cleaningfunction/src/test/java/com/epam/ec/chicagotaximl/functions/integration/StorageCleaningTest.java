/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */
package com.epam.ec.chicagotaximl.functions.integration;

import com.epam.ec.chicagotaximl.functions.StorageCleaning;
import com.google.cloud.storage.Blob;
import com.google.cloud.storage.BlobId;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import org.apache.commons.lang3.RandomStringUtils;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.truth.Truth.assertThat;

/**
 * Integration test for {@link StorageCleaning}. Access to Cloud Storage is required.
 */
@TestInstance(Lifecycle.PER_CLASS)
public class StorageCleaningTest {

    private static final String TEST_BUCKET = "chicago-taxi-ml-demo-1-test";
    private static final String TEST_FILE_1 = "files/trips/trips-00000-of-00001.csv";
    private static final String TEST_FILE_2 = "files/trips/trips-00000-of-00005.csv";

    private Storage storage;
    private List<Blob> filesToDelete = new ArrayList<>();

    @BeforeAll
    public void init() {
        storage = StorageOptions.getDefaultInstance().getService();
    }

    @Test
    public void testMoveFilesToArchive() throws Exception {
        String srcTmpDirectoryName = "testing/" + RandomStringUtils.randomAlphanumeric(7);
        String dstTmpDirectoryName = "testing/" + RandomStringUtils.randomAlphanumeric(7);

        Blob srcBlob1 = storage.get(BlobId.of(TEST_BUCKET, TEST_FILE_1))
                                .copyTo(TEST_BUCKET, srcTmpDirectoryName + "/trips-1.csv")
                                .getResult();
        filesToDelete.add(srcBlob1);
        Blob srcBlob2 = storage.get(BlobId.of(TEST_BUCKET, TEST_FILE_2))
                                .copyTo(TEST_BUCKET, srcTmpDirectoryName + "/trips-2.csv")
                                .getResult();
        filesToDelete.add(srcBlob2);

        StorageCleaning sc = new StorageCleaning(
                TEST_BUCKET, srcTmpDirectoryName + "/trips-", TEST_BUCKET, dstTmpDirectoryName);
        List<String> results = sc.moveFilesToArchive();

        assertThat(results.size()).isEqualTo(2);

        Blob dstBlob1 = storage.get(BlobId.of(TEST_BUCKET, results.get(0)));
        filesToDelete.add(dstBlob1);
        Blob dstBlob2 = storage.get(BlobId.of(TEST_BUCKET, results.get(1)));
        filesToDelete.add(dstBlob2);

        assertThat(srcBlob1.exists()).isFalse();
        assertThat(srcBlob2.exists()).isFalse();

        assertThat(dstBlob1.exists()).isTrue();
        assertThat(dstBlob2.exists()).isTrue();
    }

    @AfterAll
    public void clean() {
        for (Blob blob : filesToDelete) {
            blob.delete();
        }
    }
}
