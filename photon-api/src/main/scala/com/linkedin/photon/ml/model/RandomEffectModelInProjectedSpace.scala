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
//package com.linkedin.photon.ml.model
//
//import org.apache.spark.rdd.RDD
//import org.apache.spark.storage.StorageLevel
//import org.dmg.pmml.GeneralRegressionModel.ModelType
//
//import com.linkedin.photon.ml.TaskType.TaskType
//import com.linkedin.photon.ml.Types.{FeatureShardId, REType, REId}
//import com.linkedin.photon.ml.projector.RandomEffectProjector
//import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
//
///**
// * Representation of a random effect model in projected space.
// *
// * @param modelsInProjectedSpaceRDD The underlying models with coefficients in projected space
// * @param randomEffectProjector The projector between the original and projected spaces
// * @param randomEffectType The random effect type
// * @param featureShardId The feature shard id
// */
//protected[ml] class RandomEffectModelInProjectedSpace(
//    override protected val modelsRDD: RDD[(REId, GeneralizedLinearModel)],
//    override val randomEffectType: REType,
//    override val featureShardId: FeatureShardId,
//    val randomEffectProjector: RandomEffectProjector)
//  extends RandomEffectModel(
//    modelsRDD,
//    randomEffectType,
//    featureShardId) {
//
//  //
//  // RandomEffectModel functions
//  //
//
//  /**
//   *
//   * @return
//   */
//  override def getModels: RDD[(REId, GeneralizedLinearModel)] = randomEffectProjector.projectCoefficientsRDD(modelsRDD)
//
//  /**
//   *
//   * @param updatedModelsRdd The new sub-models with coefficients in projected space, one per random
//   *                                         effect ID
//   * @return The updated projected random effect model
//   */
//  override def update(
//      updatedModelsRdd: RDD[(REId, GeneralizedLinearModel)]): RandomEffectModelInProjectedSpace = {
//
//    val currType = this.modelType
//
//    new RandomEffectModelInProjectedSpace(
//        updatedModelsRdd,
//        randomEffectType,
//        featureShardId,
//        randomEffectProjector) {
//
//      // TODO: The model types don't necessarily match, but checking each time is slow so copy the type for now
//      override lazy val modelType: TaskType = currType
//    }
//  }
//
//  //
//  // Summarizable functions
//  //
//
//  /**
//   * Summarize this model in text format.
//   *
//   * @return A model summary in text format.
//   */
//  override def toSummaryString: String = {
//
//    val stringBuilder = new StringBuilder("Projected Random Effect Model:")
//
//    stringBuilder.append(s"\nProjected by projector class '${randomEffectProjector.getClass.getSimpleName}'")
//    summaryHelper(stringBuilder)
//
//    stringBuilder.toString()
//  }
//
//  //
//  // RDDLike functions
//  //
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//  /**
//   *
//   * @param storageLevel The storage level
//   * @return This object with all its RDDs' storage level set
//   */
//  override def persistRDD(storageLevel: StorageLevel): RandomEffectModelInProjectedSpace = {
//
//    super.persistRDD(storageLevel)
//    if (!modelsInProjectedSpaceRDD.getStorageLevel.isValid) modelsInProjectedSpaceRDD.persist(storageLevel)
//
//    this
//  }
//
//  /**
//   *
//   * @return This object with all its RDDs unpersisted
//   */
//  override def unpersistRDD(): RandomEffectModelInProjectedSpace = {
//
//    if (modelsInProjectedSpaceRDD.getStorageLevel.isValid) modelsInProjectedSpaceRDD.unpersist()
//    super.unpersistRDD()
//
//    this
//  }
//
//  /**
//   *
//   * @param name The parent name for the model RDD in this class
//   * @return This object with all its RDDs' name assigned
//   */
//  override def setName(name: String): RandomEffectModelInProjectedSpace = {
//
//    super.setName(name)
//    modelsInProjectedSpaceRDD.setName(name + " (projected)")
//
//    this
//  }
//
//  /**
//   *
//   * @return This object with all its RDDs materialized
//   */
//  override def materialize(): RandomEffectModelInProjectedSpace = {
//
//    super.materialize()
//    materializeOnce(modelsInProjectedSpaceRDD)
//
//    this
//  }
//}
