package com.epam.gcp.chicagotaximl.cleanupfunction;

import java.util.Date;

/**
 * Cloud Functions populates this object with an event information.
 */
public class GcsEvent {

  private String bucket;
  private String name;
  private String metageneration;
  private Date timeCreated;
  private Date updated;

  public String getBucket() {
    return bucket;
  }

  public void setBucket(String bucket) {
    this.bucket = bucket;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public String getMetageneration() {
    return metageneration;
  }

  public void setMetageneration(String metageneration) {
    this.metageneration = metageneration;
  }

  public Date getTimeCreated() {
    return timeCreated;
  }

  public void setTimeCreated(Date timeCreated) {
    this.timeCreated = timeCreated;
  }

  public Date getUpdated() {
    return updated;
  }

  public void setUpdated(Date updated) {
    this.updated = updated;
  }
}
