/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.data.avro

import java.util.{List => JList}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

import org.apache.avro.generic.GenericRecord
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.data.avro.NameAndTermFeatureSetContainer.{FeatureBag, NameAndTerm}
import com.linkedin.photon.ml.util.{IOUtils, Utils}

/**
 * A wrapper class for a map of feature bags to a set of features contained within that bag.
 *
 * @param nameAndTermFeatureSets A [[Map]] of [[FeatureBag]] to [[NameAndTerm]] feature sets
 */
protected[ml] class NameAndTermFeatureSetContainer private (nameAndTermFeatureSets: Map[FeatureBag, Set[NameAndTerm]]) {

  import NameAndTermFeatureSetContainer._

  /**
   * Get a map of feature key to index for the set of all [[NameAndTerm]] features in the specified feature bags.
   *
   * @param featureSectionKeys The feature bags for which to generate an index map
   * @param isAddingIntercept Whether to add a dummy feature to the generated index map for the intercept term
   * @return A map from feature key to index
   */
  def getIndexMap(featureSectionKeys: Set[FeatureBag], isAddingIntercept: Boolean): Map[String, Int] = {

    val featureNameAndTermToIndexMap = nameAndTermFeatureSets
      .filterKeys(featureSectionKeys.contains)
      .values
      .fold(Set[NameAndTerm]())(_ ++ _)
      .map { case (name, term) =>
        Utils.getFeatureKey(name, term)
      }
      .zipWithIndex
      .toMap

    if (isAddingIntercept) {
      featureNameAndTermToIndexMap + (Constants.INTERCEPT_KEY -> featureNameAndTermToIndexMap.size)
    } else {
      featureNameAndTermToIndexMap
    }
  }

  /**
   * Write the feature maps to HDFS as text files.
   *
   * @param sc The Spark context
   * @param outputDir The HDFS directory to write the feature sets as text files
   */
  def saveAsTextFiles(sc: SparkContext, outputDir: String): Unit =
    nameAndTermFeatureSets.foreach { case (featureSectionKey, featureSet) =>
      saveNameAndTermFeatureSetToTextFile(
        featureSet,
        new Path(outputDir, featureSectionKey),
        sc.hadoopConfiguration)
    }
}

object NameAndTermFeatureSetContainer {

  type NameAndTerm = (String, String)
  type FeatureBag = String

  val TEXT_DELIMITER = "\t"

  /**
   * Generate a [[NameAndTermFeatureSetContainer]] from a [[RDD]] of [[GenericRecord]]s.
   *
   * @param genericRecords The input [[RDD]] of [[GenericRecord]]s.
   * @param featureBags The set of feature bags to read
   * @return The generated [[NameAndTermFeatureSetContainer]]
   */
  def readFromGenericRecords(
      genericRecords: RDD[GenericRecord],
      featureBags: Set[String],
      numPartitions: Int): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureBags
      .map { featureSectionKey =>
        (featureSectionKey,
          readNameAndTermFeatureSetFromGenericRecords(genericRecords, featureSectionKey, numPartitions))
      }
      .toMap

    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  /**
   * Parse a set of [[NameAndTerm]] features from a [[RDD]] of feature bags (stored in Avro records).
   *
   * @param genericRecords The input [[RDD]] of Avro records
   * @param featureBag The feature bag to parse in each record
   * @return A set of parsed [[NameAndTerm]] features
   */
  private def readNameAndTermFeatureSetFromGenericRecords(
      genericRecords: RDD[GenericRecord],
      featureBag: String,
      numPartitions: Int): Set[NameAndTerm] = {

    genericRecords
      .flatMap {
        _.get(featureBag) match {
          case recordList: JList[_] =>
            recordList.asScala.map {
              case record: GenericRecord =>
                readNameAndTermFromGenericRecord(record)

              case any =>
                throw new IllegalArgumentException(
                  s"Found object with class '${any.getClass}' in features list '$featureBag': expected record" +
                    s"containing 'name', 'term', and 'value' fields")

            }

          case _ =>
            throw new IllegalArgumentException(
              s"'$featureBag' is not a list (or may be null): expected list of records containing 'name', " +
                s"'term', and 'value' fields")
        }
      }
      .distinct(numPartitions)
      .collect()
      .toSet
  }

  /**
   * Read a feature composed of two parts (name and term) from an Avro record.
   *
   * @param record The input Avro record
   * @return The parsed feature
   */
  private def readNameAndTermFromGenericRecord(record: GenericRecord): NameAndTerm = {

    val name = Utils.getStringAvro(record, AvroFieldNames.NAME)
    val term = Utils.getStringAvro(record, AvroFieldNames.TERM, isNullOK = true)

    (name, term)
  }

  /**
   * Generate a [[NameAndTermFeatureSetContainer]] from one or more text files on HDFS.
   *
   * @param path The input HDFS directory
   * @param featureBags The set of feature bags to load from the input directory
   * @param configuration The Hadoop configuration
   * @return This [[NameAndTermFeatureSetContainer]] parsed from text files on HDFS
   */
  protected[ml] def readFromTextFiles(
    path: Path,
    featureBags: Set[String],
    configuration: Configuration): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureBags
      .map { featureSectionKey =>
        val inputPath = new Path(path, featureSectionKey)
        val nameAndTermFeatureSet = readNameAndTermFeatureSetFromTextFile(inputPath, configuration)

        (featureSectionKey, nameAndTermFeatureSet)
      }
      .toMap

    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  /**
   * Read a [[Set]] of [[NameAndTerm]] from the text files within the input path.
   *
   * @param inputPath The input path
   * @param configuration the Hadoop configuration
   * @return The [[Set]] of [[NameAndTerm]] read from the text files of the given input path
   */
  private def readNameAndTermFeatureSetFromTextFile(inputPath: Path, configuration: Configuration): Set[NameAndTerm] =
    IOUtils
      .readStringsFromHDFS(inputPath, configuration)
      .map { feature =>
        feature.split(TEXT_DELIMITER) match {
          case Array(name, term) =>
            (name, term)

          case Array(name) =>
            (name, "")

          case other =>
            throw new UnsupportedOperationException(
              s"Error parsing feature '$feature' from '$inputPath': expected 1 or 2 tokens but found ${other.length}")
        }
      }
      .toSet

  /**
   * Write a [[Set]] of [[NameAndTerm]] features to HDFS as text files.
   *
   * @param nameAndTermFeatureSet The map to be written
   * @param outputPath The HDFS path to which to write the features
   * @param configuration The Hadoop configuration
   */
  private def saveNameAndTermFeatureSetToTextFile(
      nameAndTermFeatureSet: Set[NameAndTerm],
      outputPath: Path,
      configuration: Configuration): Unit = {

    val iterator = nameAndTermFeatureSet
      .iterator
      .map { case (name, term) =>
        name + TEXT_DELIMITER + term
      }

    IOUtils.writeStringsToHDFS(iterator, outputPath, configuration, forceOverwrite = false)
  }
}
