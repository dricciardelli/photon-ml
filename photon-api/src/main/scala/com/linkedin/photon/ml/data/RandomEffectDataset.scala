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
package com.linkedin.photon.ml.data

import scala.util.hashing.byteswap64

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}

import com.linkedin.photon.ml.Types.{REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.spark.RDDLike

// TODO - Documentation

/**
 * Dataset implementation for random effect data.
 *
 * All of the training data for a single random effect must fit into on Spark partition. The size limit of a single
 * Spark partition is 2 GB. If the size of (samples * features) exceeds the maximum size of a single Spark partition,
 * the data is split into two sections: active and passive data.
 *
 *   activeData + passiveData = full data set
 *
 * Active data is used for both training and scoring (to determine residuals for partial score). Passive data is used
 * only for scoring. In the vast majority of cases, all data is active data.
 *
 * @param activeData Per-entity datasets used to train per-entity models and to compute residuals
 * @param passiveData Per-entity datasets used only to compute residuals
 * @param uniqueIdToRandomEffectIds Map of unique sample id to random effect id
 */
protected[ml] class RandomEffectDataset(
    val activeData: ActiveData,
    val passiveData: PassiveData, // To Opt or not to Opt
    uniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)])
  extends Dataset[RandomEffectDataset]
    with RDDLike {

  val randomEffectIdPartitioner: Partitioner = activeData.partitioner.get
  val uniqueIdPartitioner: Partitioner = uniqueIdToRandomEffectIds.partitioner.get

  //
  // RandomEffectDataset functions
  //

  /**
   *
   * @param scores
   * @return
   */
  protected def addScoresToActiveDataOffsets(scores: CoordinateDataScores): ActiveData = {

    val scoresGroupedByRandomEffectId = scores
      .scoresRdd
      .join(uniqueIdToRandomEffectIds)
      .map { case (uniqueId, (score, reId)) => (reId, (uniqueId, score)) }
      .groupByKey(randomEffectIdPartitioner)
      .mapValues(_.toArray.sortBy(_._1))

    // Both RDDs use the same partitioner
    activeData
      .join(scoresGroupedByRandomEffectId)
      .mapValues { case (localData, localScore) => localData.addScoresToOffsets(localScore) }
  }

  /**
   *
   * @param scores
   * @return
   */
  protected def addScoresToPassiveDataOffsets(scores: CoordinateDataScores): PassiveData =
    passiveData
      .join(scores.scoresRdd)
      .mapValues { case ((randomEffectId, LabeledPoint(response, features, offset, weight)), score) =>
        (randomEffectId, LabeledPoint(response, features, offset + score, weight))
      }

  //
  // Dataset functions
  //

  /**
   * Add residual scores to the data offsets.
   *
   * @param scores The residual scores
   * @return The dataset with updated offsets
   */
  override def addScoresToOffsets(scores: CoordinateDataScores): RandomEffectDataset = {

    val updatedActiveData = addScoresToActiveDataOffsets(scores)
    val updatedPassiveDataOption = addScoresToPassiveDataOffsets(scores)

    new RandomEffectDataset(updatedActiveData, updatedPassiveDataOption, uniqueIdToRandomEffectIds)
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = activeData.sparkContext

  /**
   * Assign a given name to [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] assigned
   */
  override def setName(name: String): RandomEffectDataset = {

    activeData.setName(s"$name - Active Data")
    passiveData.setName(s"$name - Passive Data")
    uniqueIdToRandomEffectIds.setName(s"$name - UID to REID")

    this
  }

  /**
   * Set the storage level of [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]], and persist
   * their values across the cluster the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]]
   *         set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectDataset = {

    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)
    if (!uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.persist(storageLevel)

    this
  }

  /**
   * Mark [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] as non-persistent, and remove all
   * blocks for them from memory and disk.
   *
   * @return This object with [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] marked
   *         non-persistent
   */
  override def unpersistRDD(): RandomEffectDataset = {

    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    if (passiveData.getStorageLevel.isValid) passiveData.unpersist()
    if (uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.unpersist()

    this
  }

  /**
   * Materialize [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] (Spark [[RDD]]s are lazy
   * evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] materialized
   */
  override def materialize(): RandomEffectDataset = {

    materializeOnce(activeData, uniqueIdToRandomEffectIds, passiveData)

    this
  }

  //
  // Summarizable Functions
  //

  /**
   * Build a human-readable summary for [[RandomEffectDataset]].
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {

    val stringBuilder = new StringBuilder("Random Effect Data Set:")

    val activeDataValues = activeData.values.persist(StorageLevel.MEMORY_ONLY_SER)

    val numActiveSamples = uniqueIdToRandomEffectIds.count()
    val activeSampleWeighSum = activeDataValues.map(_.getWeights.map(_._2).sum).sum()
    val activeSampleResponseSum = activeDataValues.map(_.getLabels.map(_._2).sum).sum()
    val numPassiveSamples = passiveData.count()
    val passiveSampleResponsesSum = passiveData.values.map(_._2.label).sum()
    val numAllSamples = numActiveSamples + numPassiveSamples
    val numActiveSamplesStats = activeDataValues.map(_.numDataPoints).stats()
    val activeSamplerResponseSumStats = activeDataValues.map(_.getLabels.map(_._2).sum).stats()
    val numFeaturesStats = activeDataValues.map(_.numActiveFeatures).stats()

    activeDataValues.unpersist()

    stringBuilder.append(s"\nnumActiveSamples: $numActiveSamples")
    stringBuilder.append(s"\nactiveSampleWeighSum: $activeSampleWeighSum")
    stringBuilder.append(s"\nactiveSampleResponseSum: $activeSampleResponseSum")
    stringBuilder.append(s"\nnumPassiveSamples: $numPassiveSamples")
    stringBuilder.append(s"\npassiveSampleResponsesSum: $passiveSampleResponsesSum")
    stringBuilder.append(s"\nnumAllSamples: $numAllSamples")
    stringBuilder.append(s"\nnumActiveSamplesStats: $numActiveSamplesStats")
    stringBuilder.append(s"\nactiveSamplerResponseSumStats: $activeSamplerResponseSumStats")
    stringBuilder.append(s"\nnumFeaturesStats: $numFeaturesStats")

    stringBuilder.toString()
  }
}

object RandomEffectDataset {

  type ActiveData = RDD[(REId, LocalDataset)]
  type PassiveData = RDD[(UniqueSampleId, (REId, LabeledPoint))]

  /**
   * Build the random effect dataset with the given configuration.
   *
   * @param gameDataset The RDD of [[GameDatum]] used to generate the random effect dataset
   * @param randomEffectDataConfiguration The data configuration for the random effect dataset
   * @param randomEffectPartitioner The per random effect partitioner used to generated the grouped active data
   * @param existingModelKeysRddOpt
   * @return A new random effect dataset with the given configuration
   */
  def apply(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RandomEffectDataset = {

    val activeData = generateActiveData(
      gameDataset,
      randomEffectDataConfiguration,
      randomEffectPartitioner,
      existingModelKeysRddOpt)
    val passiveData = generatePassiveData(gameDataset, activeData, randomEffectDataConfiguration)
    val uniqueIdToRandomEffectIds = generateIdMap(activeData, gameDataset.partitioner.get)

    new RandomEffectDataset(activeData, passiveData, uniqueIdToRandomEffectIds)
  }

  /**
   * Generate active data.
   *
   * @param gameDataset The input dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @param randomEffectPartitioner A random effect partitioner
   * @return The active dataset
   */
  private def generateActiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): ActiveData = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    // Generate random effect data from raw GAME data for selected shard
    val keyedRandomEffectDataset = gameDataset.map { case (uniqueId, gameData) =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)

      (randomEffectId, (uniqueId, labeledPoint))
    }

    // Filter data using reservoir sampling if active data size is bounded
    val groupedRandomEffectDataset = randomEffectDataConfiguration
      .numActiveDataPointsUpperBound
      .map { activeDataUpperBound =>
        groupDataByKeyAndSample(
          keyedRandomEffectDataset,
          randomEffectPartitioner,
          activeDataUpperBound,
          randomEffectType)
      }
      .getOrElse(keyedRandomEffectDataset.groupByKey(randomEffectPartitioner))

    val filteredRandomEffectDataset = lowerBoundFilteringOnActiveData(
      groupedRandomEffectDataset,
      randomEffectDataConfiguration,
      existingModelKeysRddOpt)
    val rawActiveData = filteredRandomEffectDataset.mapValues { iterable =>
      LocalDataset(iterable.toArray, isSortedByFirstIndex = false)
    }

    // Filter features if feature dimension of active data is bounded
    featureSelectionOnActiveData(rawActiveData, randomEffectDataConfiguration)
  }

  /**
   * Generate a dataset, grouped by random effect ID and limited to a maximum number of samples selected via reservoir
   * sampling.
   *
   * The 'Min Heap' reservoir sampling algorithm is used for two reasons:
   * 1. The exact sampling must be reproducible so that [[RDD]] partitions can be recovered
   * 2. The linear algorithm is non-trivial to combine in a distributed manner
   *
   * @param rawKeyedDataset The raw dataset, with samples keyed by random effect ID
   * @param partitioner The partitioner
   * @param sampleCap The sample cap
   * @param randomEffectType The type of random effect
   * @return An RDD of data grouped by individual ID
   */
  private def groupDataByKeyAndSample(
      rawKeyedDataset: RDD[(REId, (UniqueSampleId, LabeledPoint))],
      partitioner: Partitioner,
      sampleCap: Int,
      randomEffectType: REType): RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])] = {

    // Helper class for defining a constant ordering between data samples (necessary for RDD re-computation)
    case class ComparableLabeledPointWithId(comparableKey: Int, uniqueId: UniqueSampleId, labeledPoint: LabeledPoint)
      extends Comparable[ComparableLabeledPointWithId] {

      override def compareTo(comparableLabeledPointWithId: ComparableLabeledPointWithId): Int = {
        if (comparableKey - comparableLabeledPointWithId.comparableKey > 0) {
          1
        } else {
          -1
        }
      }
    }

    val createCombiner =
      (comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
        new MinHeapWithFixedCapacity[ComparableLabeledPointWithId](sampleCap) += comparableLabeledPointWithId
      }

    val mergeValue = (
        minHeapWithFixedCapacity: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
        comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
      minHeapWithFixedCapacity += comparableLabeledPointWithId
    }

    val mergeCombiners = (
        minHeapWithFixedCapacity1: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
        minHeapWithFixedCapacity2: MinHeapWithFixedCapacity[ComparableLabeledPointWithId]) => {
      minHeapWithFixedCapacity1 ++= minHeapWithFixedCapacity2
    }

    // The reservoir sampling algorithm is fault tolerant, assuming that the uniqueId for a sample is recovered after
    // node failure. We attempt to maximize the likelihood of successful recovery through RDD replication, however there
    // is a non-zero possibility of massive failure. If this becomes an issue, we may need to resort to check-pointing
    // the raw data RDD after uniqueId assignment.
    rawKeyedDataset
      .mapValues { case (uniqueId, labeledPoint) =>
        val comparableKey = (byteswap64(randomEffectType.hashCode) ^ byteswap64(uniqueId)).hashCode()
        ComparableLabeledPointWithId(comparableKey, uniqueId, labeledPoint)
      }
      .combineByKey[MinHeapWithFixedCapacity[ComparableLabeledPointWithId]](
        createCombiner,
        mergeValue,
        mergeCombiners,
        partitioner)
      .mapValues { minHeapWithFixedCapacity =>
        val count = minHeapWithFixedCapacity.getCount
        val data = minHeapWithFixedCapacity.getData
        val weightMultiplierOpt = if (count > sampleCap) Some(1D * count / sampleCap) else None
        val dataPoints =
          data.map { case ComparableLabeledPointWithId(_, uniqueId, LabeledPoint(label, features, offset, weight)) =>
            (uniqueId, LabeledPoint(label, features, offset, weightMultiplierOpt.map(_ * weight).getOrElse(weight)))
          }
        dataPoints
      }
  }

  /**
   *
   * @param groupedData
   * @param randomEffectDataConfiguration
   * @param existingModelKeysRddOpt
   * @return
   */
  private def lowerBoundFilteringOnActiveData(
      groupedData: RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      existingModelKeysRddOpt: Option[RDD[REId]]): RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])] =
    randomEffectDataConfiguration
      .numActiveDataPointsLowerBound
      .map { activeDataLowerBound =>
        existingModelKeysRddOpt match {
          case Some(existingModelKeysRdd) =>
            groupedData.zipPartitions(existingModelKeysRdd, preservesPartitioning = true) { (dataIt, existingKeysIt) =>

              val lookupTable = existingKeysIt.toSet

              dataIt.filter { case (key, data) =>
                (data.size >= activeDataLowerBound) || !lookupTable.contains(key)
              }
            }

          case None =>
            groupedData.filter { case (_, data) =>
              data.size >= activeDataLowerBound
            }
        }
      }
      .getOrElse(groupedData)

  /**
   * Reduce active data feature dimension for individuals with few samples. The maximum feature dimension is limited to
   * the number of samples multiplied by the feature dimension ratio. Features are chosen by greatest Pearson
   * correlation score.
   *
   * @param activeData The active dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @return The active data with the feature dimension reduced to the maximum
   */
  private def featureSelectionOnActiveData(
      activeData: RDD[(REId, LocalDataset)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration): ActiveData =
    randomEffectDataConfiguration
      .numFeaturesToSamplesRatioUpperBound
      .map { numFeaturesToSamplesRatioUpperBound =>
        activeData.mapValues { localDataset =>

          var numFeaturesToKeep = math.ceil(numFeaturesToSamplesRatioUpperBound * localDataset.numDataPoints).toInt
          // In case the above product overflows
          if (numFeaturesToKeep < 0) numFeaturesToKeep = Int.MaxValue

          localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep)
        }
      }
      .getOrElse(activeData)

  /**
   *
   * @param activeData
   * @param partitioner
   * @return
   */
  private def generateIdMap(activeData: ActiveData, partitioner: Partitioner): RDD[(UniqueSampleId, REId)] =
    activeData
      .flatMap { case (individualId, localDataset) =>
        localDataset.getUniqueIds.map((_, individualId))
      }
      .partitionBy(partitioner)

  /**
   * Generate passive data set.
   *
   * @param gameDataset The raw input data set
   * @param activeData The active data set
   * @param randomEffectDataConfiguration
   * @return The passive data set
   */
  private def generatePassiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      activeData: ActiveData,
      randomEffectDataConfiguration: RandomEffectDataConfiguration): RDD[(UniqueSampleId, (REId, LabeledPoint))] = {

    // Once the active data is determined, the remaining data is considered for passive data
    val activeDataUniqueIds = activeData.flatMapValues(_.dataPoints.map(_._1)).map(_.swap)
    val keyedRandomEffectDataset = gameDataset.mapValues { gameData =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectDataConfiguration.randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(randomEffectDataConfiguration.featureShardId)

      (randomEffectId, labeledPoint)
    }

    // Calls to subtractByKey will by default use the current partitioner (which in this case is the gameDataset
    // partitioner)
    keyedRandomEffectDataset.subtractByKey(activeDataUniqueIds)

//    val passiveData = keyedRandomEffectDataset.subtractByKey(activeDataUniqueIds)
//    val passiveDataRandomEffectIds: Set[REId] = passiveData
//      .map { case (_, (randomEffectId, _)) =>
//        randomEffectId
//      }
//      .distinct()
//      .collect()
//      .toSet
//    val passiveDataRandomEffectIdsBroadcast = gameDataset.sparkContext.broadcast(passiveDataRandomEffectIds)
//
//    (passiveData, passiveDataRandomEffectIdsBroadcast)
  }
}
