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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.{REId, UniqueSampleId}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem
import com.linkedin.photon.ml.optimization.{OptimizationTracker, RandomEffectOptimizationTracker}

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The random effect optimization problem
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    protected val dataset: RandomEffectDataset,
    protected val optimizationProblem: RandomEffectOptimizationProblem[Objective])
  extends Coordinate[RandomEffectDataset](dataset) {


//  /**
//   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
//   * a starting point.
//   *
//   * @param model The model to use as a starting point
//   * @return A tuple of the updated model and the optimization states tracker
//   */
//  protected[algorithm] def updateModel(model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker])
//
//  /**
//   * Compute the regularization term value of the coordinate for a given model.
//   *
//   * @param model The model
//   * @return The regularization term value
//   */
//  protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double

  /**
   * Score the effect-specific dataset in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case randomEffectModel: RandomEffectModel =>
      RandomEffectCoordinate.score(dataset, randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(
        s"Cannot score ${dataset.getClass} using ${model.getClass} in ${this.getClass}")
  }

  /**
   * Initialize a basic model for scoring GAME data.
   *
   * @param seed A random seed
   * @return The basic model
   */
  override protected[algorithm] def initializeModel(seed: Long): RandomEffectModel =
    RandomEffectCoordinate.initializeModel(dataset, optimizationProblem)

  /**
   * Update the coordinate with a new dataset.
   *
   * @param updatedRandomEffectDataset The updated dataset
   * @return A new coordinate with the updated dataset
   */
  override protected[algorithm] def updateCoordinateWithDataset(
      updatedRandomEffectDataset: RandomEffectDataset): RandomEffectCoordinate[Objective] =
    new RandomEffectCoordinate(updatedRandomEffectDataset, optimizationProblem)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def updateModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.updateModel(dataset, optimizationProblem, randomEffectModel)

      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported")
    }

  /**
   * Compute the regularization term value of the coordinate for a given model.
   *
   * @param model The model
   * @return The regularization term value
   */
  override protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double = model match {
    case randomEffectModel: RandomEffectModel =>
      optimizationProblem.getRegularizationTermValue(randomEffectModel.modelsRDD)

    case _ =>
      throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
          s"type ${model.getClass} in ${this.getClass} is not supported")
  }
}

object RandomEffectCoordinate {

  /**
   * Score a data set using a given model.
   *
   * For information about the differences between active and passive data, see the [[RandomEffectDataset]]
   * documentation.
   *
   * @note The score is the raw dot product of the model coefficients and the feature values - it does not go through a
   *       non-linear link function.
   * @param randomEffectDataset The active data set to score
   * @param randomEffectModel The model to score the data set with
   * @return The computed scores
   */
  protected[algorithm] def score(
      randomEffectDataset: RandomEffectDataset,
      randomEffectModel: RandomEffectModel): CoordinateDataScores = {

    val activeScores = randomEffectDataset
      .activeData
      .join(randomEffectModel.modelsRDD)
      .flatMap { case (_, (localDataset, model)) =>
        localDataset.dataPoints.map { case (uniqueId, labeledPoint) =>
          (uniqueId, model.computeScore(labeledPoint.features))
        }
      }
      .partitionBy(randomEffectDataset.uniqueIdPartitioner)
      .setName("Active scores")
      .persist(StorageLevel.DISK_ONLY)

    val passiveScores = computePassiveScores(
      randomEffectDataset.passiveData,
      randomEffectDataset.passiveDataRandomEffectIds,
      randomEffectModel.modelsRDD)
      .setName("Passive scores")
      .persist(StorageLevel.DISK_ONLY)

    new CoordinateDataScores(activeScores ++ passiveScores)
  }

  /**
   * Initialize a basic model (one that has a zero model for each random effect).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataset The data set
   * @param randomEffectOptimizationProblem The optimization problem to use for creating the underlying models
   * @return A random effect model for scoring GAME data
   */
  private def initializeModel[Function <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[Function]): RandomEffectModel = {

    // TODO: Get this working for both versions of Coordinate

    val glm = randomEffectOptimizationProblem.initializeModel(0)
    val randomEffectModelsRDD = randomEffectDataset
      .activeData
      .mapValues { localDataset =>
        glm.updateCoefficients(Coefficients.initializeZeroCoefficients(localDataset.numFeatures))
      }
    val randomEffectType = randomEffectDataset.randomEffectType
    val featureShardId = randomEffectDataset.featureShardId
    val randomEffectProjector = randomEffectDataset.randomEffectProjector

    new RandomEffectModelInProjectedSpace(
      randomEffectModelsRDD,
      randomEffectProjector,
      randomEffectType,
      featureShardId)
  }

  /**
   * Update the model (i.e. run the coordinate optimizer).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataset The training dataset
   * @param randomEffectOptimizationProblem The random effect optimization problem
   * @param randomEffectModel The current model, used as a starting point
   * @return A tuple of optimized model and optimization tracker
   */
  protected[algorithm] def updateModel[Function <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[Function],
      randomEffectModel: RandomEffectModel): (RandomEffectModel, Option[RandomEffectOptimizationTracker]) = {

    // All 3 RDDs involved in these joins use the same partitioner
    val dataAndOptimizationProblems = randomEffectDataset
      .activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)

    // Left join the models to data and optimization problems for cases where we have a prior model but no new data
    val updatedModelsAndTrackers = randomEffectModel
      .modelsRDD
      .leftOuterJoin(dataAndOptimizationProblems)
      .mapValues {
        case (localModel, Some((localDataset, optimizationProblem))) =>
          val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
          val updatedModel = optimizationProblem.run(trainingLabeledPoints, localModel)
          val stateTrackers = optimizationProblem.getStatesTracker

          (updatedModel, stateTrackers)

        case (localModel, _) =>
          (localModel, None)
      }
      .setName(s"Updated models and state trackers for random effect ${randomEffectDataset.randomEffectType}")
      .persist(StorageLevel.MEMORY_ONLY)

    val updatedRandomEffectModel = randomEffectModel
      .update(updatedModelsAndTrackers.mapValues(_._1))
      .setName(s"Updated models for random effect ${randomEffectDataset.randomEffectType}")
      .persistRDD(StorageLevel.DISK_ONLY)
      .materialize()

    val optimizationTracker: Option[RandomEffectOptimizationTracker] =
      if (randomEffectOptimizationProblem.isTrackingState) {
        val stateTrackers = updatedModelsAndTrackers.flatMap(_._2._2)
        val randomEffectTracker = new RandomEffectOptimizationTracker(stateTrackers)
          .setName(s"State trackers for random effect ${randomEffectDataset.randomEffectType}")
          .persistRDD(StorageLevel.DISK_ONLY)
          .materialize()

        Some(randomEffectTracker)
      } else {
        None
      }

    updatedModelsAndTrackers.unpersist()

    (updatedRandomEffectModel, optimizationTracker)
  }

  /**
   * Score the passive data of a dataset.
   *
   * For information about the differences between active and passive data, see the [[RandomEffectDataset]]
   * documentation.
   *
   * @param passiveData The passive data to score
   * @param passiveDataRandomEffectIds The set of random effect ids with passive data
   * @param randomEffectModel The model to score the data set with
   * @return The scores computed using the models
   */
  private def scorePassiveData(
      passiveData: RDD[(UniqueSampleId, (REId, LabeledPoint))],
      passiveDataRandomEffectIds: Broadcast[Set[REId]],
      randomEffectModel: RandomEffectModel): RDD[(UniqueSampleId, Double)] = {

    val modelsForPassiveData = randomEffectModel
      .modelsRDD
      .filter { case (reId, _) =>
        passiveDataRandomEffectIds.value.contains(reId)
      }
      .collectAsMap()

    // TODO: Need a better design that properly unpersists the broadcasted variables and persists the computed RDD
    val modelsForPassiveDataBroadcast = passiveData.sparkContext.broadcast(modelsForPassiveData)
    val passiveScores = passiveData.mapValues { case (randomEffectId, labeledPoint) =>
      modelsForPassiveDataBroadcast.value(randomEffectId).computeScore(labeledPoint.features)
    }

    passiveScores.setName("Passive scores").persist(StorageLevel.DISK_ONLY).count()
    modelsForPassiveDataBroadcast.unpersist()

    passiveScores
  }
}
