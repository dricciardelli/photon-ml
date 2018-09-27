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
//package com.linkedin.photon.ml.data
//
//import org.apache.spark.Partitioner
//import org.apache.spark.broadcast.Broadcast
//import org.apache.spark.rdd.RDD
//import org.apache.spark.storage.StorageLevel
//
//import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
//import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
//import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
//import com.linkedin.photon.ml.projector.{ProjectorType, RandomEffectProjector, RandomEffectProjectorFactory}
//import com.linkedin.photon.ml.spark.RDDLike
//
///**
// * A [[RandomEffectDataset]] that has been projected into another feature space; data set implementation for projected
// * random effect data.
// *
// * @param activeData Per-entity data sets used to train per-entity models and to compute residuals
// * @param uniqueIdToRandomEffectIds Map of unique sample id to random effect id
// * @param passiveDataOption Per-entity data sets used only to compute residuals
// * @param passiveDataRandomEffectIdsOption Set of IDs for random effects which have passive data
// * @param randomEffectType The random effect type (e.g. "memberId")
// * @param featureShardId The feature shard ID
// * @param randomEffectProjector The [[RandomEffectProjector]] used to project the random effect data between feature
// *                              spaces
// */
//class RandomEffectDatasetInProjectedSpace(
//    override val activeData: ActiveData,
//    override val uniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)],
//    override val passiveDataOption: Option[PassiveData],
//    override val passiveDataRandomEffectIdsOption: Option[Broadcast[Set[String]]],
//    override val randomEffectType: REType,
//    override val featureShardId: FeatureShardId,
//    val randomEffectProjector: RandomEffectProjector)
//  extends RandomEffectDataset(
//    activeData,
//    uniqueIdToRandomEffectIds,
//    passiveDataOption,
//    passiveDataRandomEffectIdsOption,
//    randomEffectType,
//    featureShardId) {
//
//  //
//  // Dataset functions
//  //
//
//  /**
//   * Add residual scores to the data offsets.
//   *
//   * @param scores The residual scores
//   * @return The dataset with updated offsets
//   */
//  override def addScoresToOffsets(scores: CoordinateDataScores): RandomEffectDatasetInProjectedSpace = {
//
//    val updatedActiveData = addScoresToActiveDataOffsets(scores)
//    val updatedPassiveDataOption = addScoresToPassiveDataOffsets(scores)
//
//    new RandomEffectDatasetInProjectedSpace(
//      updatedActiveData,
//      uniqueIdToRandomEffectIds,
//      updatedPassiveDataOption,
//      passiveDataRandomEffectIdsOption,
//      randomEffectType,
//      featureShardId,
//      randomEffectProjector)
//  }
//
//  //
//  // RDDLike Functions
//  //
//
//  /**
//   *
//   * @param name The parent name for all RDDs in this class
//   * @return This object with all its RDDs' name assigned
//   */
//  override def setName(name: String): this.type = {
//
//    super.setName(name)
//
//    randomEffectProjector match {
//      case rddLike: RDDLike => rddLike.setName(s"$name - REID Projectors")
//      case _ =>
//    }
//
//    this
//  }
//
//  /**
//   *
//   * @param storageLevel The storage level
//   * @return This object with all its RDDs' storage level set
//   */
//  override def persistRDD(storageLevel: StorageLevel): this.type = {
//
//    super.persistRDD(storageLevel)
//
//    randomEffectProjector match {
//      case rddLike: RDDLike => rddLike.persistRDD(storageLevel)
//      case _ =>
//    }
//
//    this
//  }
//
//  /**
//   *
//   * @return This object with all its RDDs unpersisted
//   */
//  override def unpersistRDD(): this.type = {
//
//    randomEffectProjector match {
//      case rddLike: RDDLike => rddLike.unpersistRDD()
//      case _ =>
//    }
//
//    super.unpersistRDD()
//
//    this
//  }
//
//  /**
//   *
//   * @return This object with all its RDDs materialized
//   */
//  override def materialize(): this.type = {
//
//    super.materialize()
//
//    randomEffectProjector match {
//      case rddLike: RDDLike => rddLike.materialize()
//      case _ =>
//    }
//
//    this
//  }
//}
//
//object RandomEffectDatasetInProjectedSpace {
//
//  /**
//   *
//   * @param gameDataset
//   * @param randomEffectDataConfiguration
//   * @param randomEffectPartitioner
//   * @param projectorType
//   * @return
//   */
//  def apply(
//      gameDataset: RDD[(UniqueSampleId, GameDatum)],
//      randomEffectDataConfiguration: RandomEffectDataConfiguration,
//      randomEffectPartitioner: Partitioner,
//      projectorType: ProjectorType): RandomEffectDatasetInProjectedSpace = {
//
//    val randomEffectType = randomEffectDataConfiguration.randomEffectType
//    val featureShardId = randomEffectDataConfiguration.featureShardId
//
//    val unprojectedActiveData = RandomEffectDataset
//      .generateActiveData(gameDataset, randomEffectDataConfiguration, randomEffectPartitioner)
//      .setName("Active Data")
//      .persist(StorageLevel.MEMORY_AND_DISK)
//    val globalIdToIndividualIds = RandomEffectDataset
//      .generateIdMap(unprojectedActiveData, gameDataset.partitioner.get)
//      .setName("UID to REID")
//      .persist(StorageLevel.DISK_ONLY)
//    val (unprojectedPassiveDataOption, passiveDataIdsOption) =
//      randomEffectDataConfiguration.numPassiveDataPointsLowerBound match {
//        case Some(passiveDataLowerBound) =>
//          val result = RandomEffectDataset.generatePassiveData(
//            gameDataset,
//            unprojectedActiveData,
//            randomEffectType,
//            featureShardId,
//            passiveDataLowerBound)
//
//          (Some(result._1), Some(result._2))
//
//        case None =>
//          (None, None)
//      }
//
//    val projector = RandomEffectProjectorFactory.build(
//      projectorType,
//      unprojectedActiveData,
//      unprojectedPassiveDataOption)
//    projector match {
//      case projectorRdd: RDDLike =>
//        projectorRdd
//          .setName("REID Projectors")
//          .persistRDD(StorageLevel.DISK_ONLY)
//
//      case _ =>
//    }
//
//    val activeData = projector
//      .projectActiveData(activeData)
//      .setName("Projected Active Data")
//      .persist(StorageLevel.DISK_ONLY)
//    val passiveDataOption = unprojectedPassiveDataOption.map { unprojectedPassiveData =>
//      projector
//        .projectPassiveData(unprojectedPassiveData, passiveDataIdsOption.get)
//        .setName("Projected Passive Data")
//        .persist(StorageLevel.DISK_ONLY)
//    }
//
//    unprojectedActiveData.unpersist()
//
//    new RandomEffectDatasetInProjectedSpace(
//      activeData,
//      globalIdToIndividualIds,
//      passiveDataOption,
//      passiveDataIdsOption,
//      randomEffectType,
//      featureShardId,
//      projector)
//  }
//}
