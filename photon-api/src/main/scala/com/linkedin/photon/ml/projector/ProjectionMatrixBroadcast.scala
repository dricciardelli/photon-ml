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
package com.linkedin.photon.ml.projector

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.spark.BroadcastLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

// TODO - Documentation

/**
 * Represents a broadcast projection matrix.
 *
 * @param projectionMatrixBroadcast The projection matrix
 */
protected[ml] class ProjectionMatrixBroadcast(projectionMatrixBroadcast: Broadcast[ProjectionMatrix])
  extends RandomEffectProjector
  with BroadcastLike
  with Serializable {

  val projectionMatrix: ProjectionMatrix = projectionMatrixBroadcast.value

  /**
   * Project the active data set from the original space to the projected space.
   *
   * @param activeData
   * @return The same active data set in the projected space
   */
  override def projectActiveData(activeData: ActiveData): ActiveData =
    activeData.mapValues(_.projectFeatures(projectionMatrixBroadcast.value))

  /**
   * Project the passive data set from the original space to the projected space.
   *
   * @param passiveData
   * @param passiveDataIds
   * @return The same passive data set in the projected space
   */
  override def projectPassiveData(passiveData: PassiveData, passiveDataIds: Broadcast[Set[REId]]): PassiveData =
    passiveData.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>

      (shardId, LabeledPoint(response, projectionMatrixBroadcast.value.projectFeatures(features), offset, weight))
    }

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the projected space back to the original
   * space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   */
  override def projectCoefficientsRDD(
      modelsRDD: RDD[(REId, GeneralizedLinearModel)]): RDD[(REId, GeneralizedLinearModel)] =

    modelsRDD.mapValues { model =>
      val oldCoefficients = model.coefficients
      model.updateCoefficients(
        Coefficients(
          projectionMatrixBroadcast.value.projectCoefficients(oldCoefficients.means),
          oldCoefficients.variancesOption))
    }

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the original space to the projected space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   */
  override def transformCoefficientsRDD(
      modelsRDD: RDD[(REId, GeneralizedLinearModel)]): RDD[(REId, GeneralizedLinearModel)] =

    modelsRDD.mapValues { model =>
      val oldCoefficients = model.coefficients
      model.updateCoefficients(
        Coefficients(
          projectionMatrixBroadcast.value.projectFeatures(oldCoefficients.means),
          oldCoefficients.variancesOption))
    }

  /**
   * Project a [[NormalizationContext]] from the original space to the projected space.
   *
   * @param originalNormalizationContext The [[NormalizationContext]] in the original space
   * @return The same [[NormalizationContext]] in projected space
   */
  def projectNormalizationContext(originalNormalizationContext: NormalizationContext): NormalizationContext = {

    val factors = originalNormalizationContext.factorsOpt.map(factors => projectionMatrix.projectFeatures(factors))
    val shiftsAndIntercept = originalNormalizationContext
      .shiftsAndInterceptOpt
      .map { case (shifts, _) =>
        (projectionMatrix.projectFeatures(shifts), projectionMatrix.projectedInterceptId)
      }

    new NormalizationContext(factors, shiftsAndIntercept)
  }

  /**
   * Asynchronously delete cached copies of the [[ProjectionMatrix]] [[Broadcast]] on the executors.
   *
   * @return This [[ProjectionMatrixBroadcast]] with its [[ProjectionMatrix]] unpersisted
   */
  override def unpersistBroadcast(): this.type = {
    projectionMatrixBroadcast.unpersist()
    this
  }
}

object ProjectionMatrixBroadcast {

  /**
   * Generate random projection based broadcast projector
   *
   * @param originalSpaceDimension The dimension of the original feature space
   * @param projectedSpaceDimension The dimension of the projected feature space
   * @param isKeepingInterceptTerm Whether to keep the intercept in the original feature space
   * @param seed The seed of random number generator
   * @return The generated random projection based broadcast projector
   */
  protected[ml] def apply(
      originalSpaceDimension: Int,
      projectedSpaceDimension: Int,
      isKeepingInterceptTerm: Boolean,
      seed: Long = MathConst.RANDOM_SEED): ProjectionMatrixBroadcast = {

    val randomProjectionMatrix = ProjectionMatrix.buildGaussianRandomProjectionMatrix(
      projectedSpaceDimension,
      originalSpaceDimension,
      isKeepingInterceptTerm,
      seed)
    val randomProjectionMatrixBroadcast = SparkSession
      .builder()
      .getOrCreate()
      .sparkContext
      .broadcast(randomProjectionMatrix)

    new ProjectionMatrixBroadcast(randomProjectionMatrixBroadcast)
  }
}
