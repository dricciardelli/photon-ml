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
package com.linkedin.photon.ml.optimization.hyperparameter

import scala.math.exp
import scala.util.Random

import breeze.linalg.DenseVector

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.function.glm.{LogisticLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.function.svm.SmoothedHingeLossFunction
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization.{OWLQN, Optimizer, RegularizationContext}
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}

/**
 * Evaluation function to use for single-node optimization problem hyper-parameter tuning. Currently, the only
 * hyper-parameter is regularization weight.
 *
 * Given a hyper-parameter vector, this evaluation function will setup an optimization problem. It will perform
 * k-fold cross-validation using the data and optimization problem, and return the mean of the evaluation results.
 *
 * @tparam Objective The type of objective function to optimize, using a single node
 * @param optimizer The optimizer to use
 * @param objectiveFunction The objective function to optimize
 * @param regularizationContext The regularization context
 * @param initialModel The initial model from which to begin optimization
 * @param input The input data over which to optimize
 * @param numFolds The number of folds in the cross-validation
 */
protected[optimization] class SingleNodeEvaluationFunction[Objective <: SingleNodeObjectiveFunction](
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    regularizationContext: RegularizationContext,
    initialModel: GeneralizedLinearModel,
    input: Iterable[LabeledPoint],
    numFolds: Int)
  extends EvaluationFunction[(Double, Double)] {

  require(numFolds > 1, s"Invalid number of folds '$numFolds'; must be at least 2")

  private val lossFunction = initialModel match {
    case _: LogisticRegressionModel => LogisticLossFunction.lossAndDzLoss _
    case _: LinearRegressionModel => SquaredLossFunction.lossAndDzLoss _
    case _: PoissonRegressionModel => PoissonLossFunction.lossAndDzLoss _
    case _: SmoothedHingeLossLinearSVMModel => SmoothedHingeLossFunction.lossAndDzLoss _
  }

  /**
   * Evaluate the optimization problem for the hyper-parameter vector on a single fold.
   *
   * @param trainInput The training data
   * @param validationInput The validation data
   * @return The log-loss of the validation data for the model optimized over the training data
   */
  private def evaluate(trainInput: Iterable[LabeledPoint], validationInput: Iterable[LabeledPoint]): Double = {

    // Train using the new regularization weight (If we're using log-loss, no point in converting to original space)
    val optimizedCoefficients = Coefficients(
      optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(trainInput)._1,
      None)

    // Score the validation set with the new model
    validationInput
      .map(x => x.weight * lossFunction(optimizedCoefficients.computeScore(x.features) + x.offset, x.label)._1)
      .sum

    // Alternate method which uses some evaluation metric
    //      val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(trainInput)
    //      val normalizationContext = optimizer.getNormalizationContext
    //      val model = createModel(normalizationContext, optimizedCoefficients, None)
    //      val scoresLabelsAndWeights =
    //        .map(x => (0L, (model.computeMean(x.features, x.offset), x.label, x.weight))
    //      evaluator.evaluateWithScoresAndLabelsAndWeights(scoresLabelsAndWeights)
  }

  /**
   * Performs the evaluation.
   *
   * @param hyperParameters The vector of hyper-parameter values under which to evaluate the function
   * @return A tuple of the evaluated value and the original output from the inner estimator
   */
  override def apply(hyperParameters: DenseVector[Double]): (Double, (Double, Double)) = {

    // Unpack and update regularization weight
    val regularizationWeight = exp(hyperParameters(0))
    objectiveFunction match {
      case func: L2Regularization =>
        func.l2RegularizationWeight = regularizationContext.getL2RegularizationWeight(regularizationWeight)
    }
    optimizer match {
      case owlqn: OWLQN =>
        owlqn.l1RegularizationWeight = regularizationContext.getL1RegularizationWeight(regularizationWeight)
    }

    // Create cross-validation folds
    val shuffledData = Random.shuffle(input)
    val foldSize = ((1D / numFolds) * input.size).toInt
    val foldData = shuffledData
      .grouped(foldSize)
      .zipWithIndex
      .map { case (dataFold, index) =>
        (index, dataFold)
      }
      .toMap
    val folds = for (i <- 0 until numFolds) yield (foldData.filterNot(_._1 == i).values, foldData(i))

    // Evaluate folds and filter poor results
    val evaluations = folds
      .map { case (trainingDataIterable, validationData) =>
        evaluate(trainingDataIterable.foldLeft(Seq[LabeledPoint]())(_ ++ _), validationData)
      }
      .filter(eval => !eval.isNaN && !eval.isInfinite)

    // Determine and return average evaluation result
    val meanEvaluation = if (evaluations.nonEmpty) {
      evaluations.sum / evaluations.length
    } else {
      Double.NegativeInfinity
    }

    (meanEvaluation, (regularizationWeight, meanEvaluation))
  }

  /**
   * Extracts a vector representation from the hyper-parameters associated with the original estimator output.
   *
   * @param result The original estimator output
   * @return A vector representation of the hyper-parameters
   */
  override def vectorizeParams(result: (Double, Double)): DenseVector[Double] = DenseVector(result._1)

  /**
   * Extracts the evaluation value from the original estimator output.
   *
   * @param result The original estimator output
   * @return The evaluation value
   */
  override def getEvaluationValue(result: (Double, Double)): Double = result._2
}

object SingleNodeEvaluationFunction {

  val DEFAULT_NUM_FOLDS = 5

  /**
   * Factory method to create new [[SingleNodeEvaluationFunction]] objects.
   *
   * @tparam Objective The type of objective function to optimize, using a single node
   * @param optimizer The optimizer to use
   * @param objectiveFunction The objective function to optimize
   * @param regularizationContext The regularization context
   * @param initialModel The initial model from which to begin optimization
   * @param input The input data over which to optimize
   * @return A new [[SingleNodeEvaluationFunction]]
   */
  protected[optimization] def apply[Objective <: SingleNodeObjectiveFunction](
      optimizer: Optimizer[Objective],
      objectiveFunction: Objective,
      regularizationContext: RegularizationContext,
      initialModel: GeneralizedLinearModel,
      input: Iterable[LabeledPoint]): SingleNodeEvaluationFunction[Objective] =
    new SingleNodeEvaluationFunction(
      optimizer,
      objectiveFunction,
      regularizationContext,
      initialModel,
      input,
      DEFAULT_NUM_FOLDS)
}
