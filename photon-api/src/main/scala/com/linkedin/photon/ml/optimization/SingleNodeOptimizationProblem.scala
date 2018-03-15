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
package com.linkedin.photon.ml.optimization

import scala.math.log

import breeze.linalg.Vector

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.hyperparameter.search.RandomSearch
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.optimization.hyperparameter.SingleNodeEvaluationFunction
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import com.linkedin.photon.ml.util.{BroadcastWrapper, DoubleRange}

/**
 * An optimization problem solved by a single task on one executor. Used for solving the per-entity optimization
 * problems of a random effect model.
 *
 * @tparam Objective The type of objective function to optimize, using a single node
 * @param optimizer The underlying optimizer which iteratively solves the convex problem
 * @param objectiveFunction The objective function to optimize
 * @param glmConstructor The function to use for producing GLMs from trained coefficients
 * @param isComputingVariances Should coefficient variances be computed in addition to the means?
 * @param seed Random seed
 */
protected[ml] class SingleNodeOptimizationProblem[Objective <: SingleNodeObjectiveFunction] protected[optimization] (
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    regularizationContext: RegularizationContext,
    isComputingVariances: Boolean,
    seed: Long = System.currentTimeMillis)
  extends GeneralizedLinearOptimizationProblem[Objective](
    optimizer,
    objectiveFunction,
    glmConstructor,
    isComputingVariances)
  with Serializable {

  import SingleNodeOptimizationProblem._

  /**
   * Compute coefficient variances
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return The feature coefficient variances
   */
  override def computeVariances(input: Iterable[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] = {
    (isComputingVariances, objectiveFunction) match {
      case (true, twiceDiffFunc: TwiceDiffFunction) =>
        Some(twiceDiffFunc
          .hessianDiagonal(input, coefficients)
          .map(v => 1.0 / (v + MathConst.EPSILON)))

      case _ =>
        None
    }
  }

  /**
   * Run the optimization algorithm on the input data, starting from an initial model of all-0 coefficients.
   *
   * @param input The training data
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  override def run(input: Iterable[LabeledPoint]): GeneralizedLinearModel =
    run(input, initializeZeroModel(input.head.features.size))

  /**
   * Runs hyper-parameter optimization to find the best regularization weight.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @param objectiveFunction The objective function
   * @param range The range of regularization weights to explore
   * @param iterations The number of hyper-parameter tuning iterations to perform
   * @return The best regularization weight for the training data and settings (if one was found)
   */
  private def runHyperParameterTuning(
      input: Iterable[LabeledPoint],
      initialModel: GeneralizedLinearModel,
      objectiveFunction: Objective,
      range: DoubleRange,
      iterations: Int): Option[Double] = {

    // Setup evaluation function
    val evaluationFunction = SingleNodeEvaluationFunction(
      optimizer,
      objectiveFunction,
      regularizationContext,
      initialModel,
      input)

    // Setup search algorithm
    val searcher = new RandomSearch[(Double, Double)](
      List(DoubleRange(log(range.start), log(range.end))),
      evaluationFunction,
      seed = seed)

    // This is hanging, for some reason. Each model train / test cycle happens really fast for these sub-problems,
    // though, so it might not be necessary since we can do a lot of evaluations
    // val searcher = new GaussianProcessSearch[(GeneralizedLinearModel, Double, Double)](
    //   List(range),
    //   evaluationFunction,
    //   evaluator,
    //   seed = seed)

    // Run hyper-parameter search
    val results = searcher.find(iterations)

    // Return best hyper-parameter
    val (bestWeight, bestEval) = results.minBy(_._2)
    if (!bestEval.isNaN && !bestEval.isInfinite) {
      Some(bestWeight)
    } else {
      None
    }
  }

  /**
   * Run the optimization algorithm on the input data, starting from the initial model provided.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  override def run(input: Iterable[LabeledPoint], initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {

    // Disabled for now
    if (false && input.size >= DEFAULT_SAMPLES_LOWER_BOUND) {
      val optimalRegWeight = runHyperParameterTuning(
        input,
        initialModel,
        objectiveFunction,
        DEFAULT_HYPER_PARAMETER_RANGE,
        DEFAULT_TUNING_ITERATIONS)

      objectiveFunction match {
        case func: L2Regularization =>
          optimalRegWeight.foreach { weight =>
            func.l2RegularizationWeight = regularizationContext.getL2RegularizationWeight(weight)
          }
      }
      optimizer match {
        case owlqn: OWLQN =>
          optimalRegWeight.foreach { weight =>
            owlqn.l1RegularizationWeight = regularizationContext.getL2RegularizationWeight(weight)
          }
      }
    }

    val normalizationContext = optimizer.getNormalizationContext
    val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(input)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    modelTrackerBuilder.foreach { modelTrackerBuilder =>
      val tracker = optimizer.getStateTracker.get
      logger.info(s"History tracker information:\n $tracker")
      val modelsPerIteration = tracker.getTrackedStates.map { x =>
        val coefficients = x.coefficients
        val variances = computeVariances(input, coefficients)
        createModel(normalizationContext, coefficients, variances)
      }
      logger.info(s"Number of iterations: ${modelsPerIteration.length}")
      modelTrackerBuilder += ModelTracker(tracker, modelsPerIteration)
    }

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }
}

object SingleNodeOptimizationProblem {

  val DEFAULT_HYPER_PARAMETER_RANGE = DoubleRange(1E-5, 1E5)
  val DEFAULT_SAMPLES_LOWER_BOUND = 100
  val DEFAULT_TUNING_ITERATIONS = 100

  /**
   * Factory method to create new [[SingleNodeOptimizationProblem]] objects.
   *
   * @tparam Function The type of objective function to optimize, using a single node
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param isTrackingState Should the optimization problem record the internal optimizer states?
   * @param isComputingVariance Should coefficient variances be computed in addition to the means?
   * @return A new [[SingleNodeOptimizationProblem]]
   */
  def apply[Function <: SingleNodeObjectiveFunction](
      configuration: GLMOptimizationConfiguration,
      objectiveFunction: Function,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      isTrackingState: Boolean,
      isComputingVariance: Boolean): SingleNodeOptimizationProblem[Function] = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    // Will result in a runtime error if created Optimizer cannot be cast to an Optimizer that can handle the given
    // objective function.
    val optimizer = OptimizerFactory
      .build(optimizerConfig, normalizationContext, regularizationContext, regularizationWeight, isTrackingState)
      .asInstanceOf[Optimizer[Function]]

    new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      glmConstructor,
      regularizationContext,
      isComputingVariance)
  }
}
