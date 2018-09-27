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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

// TODO - Documentation

/**
 *
 * @param indexMapProjectorRDD The projectors
 */
protected[ml] class IndexMapProjectorRDD private (indexMapProjectorRDD: RDD[(REId, IndexMapProjector)])
  extends RandomEffectProjector
  with RDDLike {

  /**
   * Project the active data set from the original space to the projected space.
   *
   * @param activeData
   * @return The same active data set in the projected space
   */
  override def projectActiveData(activeData: ActiveData): ActiveData =
    activeData
      // Make sure the activeData retains its partitioner, especially when the partitioner of featureMaps is
      // not the same as that of activeData
      .join(indexMapProjectorRDD, activeData.partitioner.get)
      .mapValues { case (localDataset, projector) => localDataset.projectFeatures(projector) }

  /**
   * Project the passive data set from the original space to the projected space.
   *
   * @param passiveData
   * @param passiveDataIds
   * @return The same passive data set in the projected space
   */
  override def projectPassiveData(passiveData: PassiveData, passiveDataIds: Broadcast[Set[REId]]): PassiveData = {

    val passiveDataProjectors = indexMapProjectorRDD
      .filter { case (randomEffectId, _) =>
        passiveDataIds.value.contains(randomEffectId)
      }
      .collectAsMap()

    val passiveDataProjectorsBroadcast = passiveData.sparkContext.broadcast(passiveDataProjectors)
    val result = passiveData.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
      val projector = passiveDataProjectorsBroadcast.value(shardId)

      (shardId, LabeledPoint(response, projector.projectFeatures(features), offset, weight))
    }

    passiveDataProjectorsBroadcast.unpersist()

    result
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

    // Left join the models to projectors for cases where we have a prior model but no new data (and hence no
    // projectors)
    modelsRDD
      .leftOuterJoin(indexMapProjectorRDD)
      .mapValues { case (model, projectorOpt) =>
        projectorOpt.map { projector =>
          val oldCoefficients = model.coefficients

          model.updateCoefficients(
            Coefficients(
              projector.projectCoefficients(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectCoefficients)))
        }.getOrElse(model)
      }

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the original space to the projected space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   */
  override def transformCoefficientsRDD(
      modelsRDD: RDD[(REId, GeneralizedLinearModel)]): RDD[(REId, GeneralizedLinearModel)] =

    // Left join the models to projectors for cases where we have a prior model but no new data (and hence no
    // projectors)
    modelsRDD
      .leftOuterJoin(indexMapProjectorRDD)
      .mapValues { case (model, projectorOpt) =>
        projectorOpt.map { projector =>
          val oldCoefficients = model.coefficients

          model.updateCoefficients(
            Coefficients(
              projector.projectFeatures(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectFeatures)))
        }.getOrElse(model)
      }

  /**
   * Project a [[NormalizationContext]] from the original space to the projected space.
   *
   * @param originalNormalizationContext The [[NormalizationContext]] in the original space
   * @return The same [[NormalizationContext]] in projected space
   */
  def projectNormalizationRDD(originalNormalizationContext: NormalizationContext): RDD[(REId, NormalizationContext)] =

    indexMapProjectorRDD.mapValues { projector =>
      val factors = originalNormalizationContext.factorsOpt.map(factors => projector.projectFeatures(factors))
      val shiftsAndIntercept = originalNormalizationContext
        .shiftsAndInterceptOpt
        .map { case (shifts, intercept) =>
          val newShifts = projector.projectFeatures(shifts)
          val newIntercept = projector.originalToProjectedSpaceMap(intercept)

          (newShifts, newIntercept)
        }

      new NormalizationContext(factors, shiftsAndIntercept)
    }

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = indexMapProjectorRDD.sparkContext

  /**
   * Assign a given name to [[indexMapProjectorRDD]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[indexMapProjectorRDD]] assigned
   */
  override def setName(name: String): IndexMapProjectorRDD = {

    indexMapProjectorRDD.setName(name)

    this
  }

  /**
   * Set the storage level of [[indexMapProjectorRDD]], and persist their values across the cluster the first time they are
   * computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[indexMapProjectorRDD]] set
   */
  override def persistRDD(storageLevel: StorageLevel): IndexMapProjectorRDD = {

    if (!indexMapProjectorRDD.getStorageLevel.isValid) indexMapProjectorRDD.persist(storageLevel)

    this
  }

  /**
   * Mark [[indexMapProjectorRDD]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[indexMapProjectorRDD]] marked non-persistent
   */
  override def unpersistRDD(): IndexMapProjectorRDD = {

    if (indexMapProjectorRDD.getStorageLevel.isValid) indexMapProjectorRDD.unpersist()

    this
  }

  /**
   * Materialize [[indexMapProjectorRDD]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[indexMapProjectorRDD]] materialized
   */
  override def materialize(): IndexMapProjectorRDD = {

    materializeOnce(indexMapProjectorRDD)

    this
  }
}

object IndexMapProjectorRDD {

  /**
   * Generate index map based RDD projectors.
   *
   * @param indices
   * @param originalSpaceDimension
   * @return The generated index map based RDD projectors
   */
  protected[ml] def apply(indices: RDD[(REId, Set[Int])], originalSpaceDimension: Int): IndexMapProjectorRDD = {

    val indexMapProjectors = indices.mapValues { indexSet =>
      new IndexMapProjector(indexSet.zipWithIndex.toMap, originalSpaceDimension, indexSet.size)
    }

    new IndexMapProjectorRDD(indexMapProjectors)
  }
}
