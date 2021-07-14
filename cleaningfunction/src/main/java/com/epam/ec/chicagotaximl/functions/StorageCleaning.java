/**
 * Copyright © 2021 EPAM Systems, Inc. All Rights Reserved. All information contained herein is, and remains the
 * property of EPAM Systems, Inc. and/or its suppliers and is protected by international intellectual
 * property law. Dissemination of this information or reproduction of this material is strictly forbidden,
 * unless prior written permission is obtained from EPAM Systems, Inc
 */
package com.epam.ec.chicagotaximl.functions;

import com.google.api.gax.paging.Page;
import com.google.cloud.storage.Blob;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Moves CSV files to archive.
 */
@AllArgsConstructor
public class StorageCleaning {

    private final String srcBucket;
    private final String srcLocationPrefix;
    private final String dstBucket;
    private final String dstDirectory;

    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH:mm:ss");

    public List<String> moveFilesToArchive() {
        String destinationDirectoryName = createDestinationDirectoryName(LocalDateTime.now(ZoneId.of("US/Eastern")));
        Storage storage = StorageOptions.getDefaultInstance().getService();
        Page<Blob> list = storage.list(srcBucket, Storage.BlobListOption.prefix(srcLocationPrefix));

        List<String> results = new ArrayList<>();
        for (Blob blob : list.iterateAll()) {
            if (blob.getName().endsWith(".csv")) {
                String newName = destinationDirectoryName + blob.getName();
                blob.copyTo(dstBucket, newName);
                blob.delete();
                results.add(newName);
            }
        }
        return results;
    }

    /**
     * Returns destination path. Example: trips/2020-11-23-09:10:20/
     */
    @VisibleForTesting
    String createDestinationDirectoryName(LocalDateTime dateTime) {
        return String.format("%s/%s/", dstDirectory, dateTime.format(DATE_TIME_FORMATTER));
    }
}
