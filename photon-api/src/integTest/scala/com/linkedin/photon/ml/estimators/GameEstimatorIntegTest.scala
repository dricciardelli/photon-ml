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
package com.linkedin.photon.ml.estimators

import scala.language.existentials

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.slf4j.Logger
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType._
import com.linkedin.photon.ml.Types._
import com.linkedin.photon.ml.algorithm.CoordinateDescent
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.model.{FixedEffectModel, GameModel}
import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationContextWrapper, NormalizationType}
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.optimization.{L2RegularizationContext, OptimizerConfig, OptimizerType}
import com.linkedin.photon.ml.projector.RandomProjection
import com.linkedin.photon.ml.stat.FeatureDataStatistics
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.util._

/**
 * Integration tests for [[GameEstimator]].
 */
class GameEstimatorIntegTest extends SparkTestUtils with GameTestUtils {

  import GameEstimatorIntegTest._

  /**
   * A very simple test that fits a toy dataset using only the [[GameEstimator]] (not the full Driver).
   * This is useful to understand the minimum setting in which a [[GameEstimator]] will function properly,
   * and to verify the algorithms manually.
   *
   * @note This is a very good pedagogical example that uses a minimal setup to run the GameEstimator.
   * @note Intercepts are optional in [[GameEstimator]]. Here, we have to setup an intercept manually, otherwise
   *       [[GameEstimator]] learns only a dependence on the features.
   */
  @Test(dataProvider = "trivialLabeledPoints")
  def simpleHardcodedTest(labeledPoints: Seq[LabeledPoint]): Unit = sparkTest("simpleHardcodedTest") {

    val (coordinateId, featureShardId) = ("global", "features")

    val trainingDataSeq = labeledPoints
      .zipWithIndex
      .map { case(point, index) =>
        val newPoint = LabeledPoint(point.label, new DenseVector[Double](point.features.toArray :+ 1.0), 0D, 1D)
        (index.toLong, newPoint)
      }
    val trainingDataRdd = sc.parallelize(trainingDataSeq)
    val fixedEffectDataset = new FixedEffectDataset(trainingDataRdd, featureShardId)
    val trainingDatasets = Map((coordinateId, fixedEffectDataset))

    val fixedEffectOptConfig = FixedEffectOptimizationConfiguration(
      OptimizerConfig(
        optimizerType = OptimizerType.LBFGS,
        maximumIterations = 100,
        tolerance = 1e-11,
        constraintMap = None),
      L2RegularizationContext,
      regularizationWeight = 0.3)
    val modelConfig: GameEstimator.GameOptimizationConfiguration = Map((coordinateId, fixedEffectOptConfig))
    val logger = createLogger("SimpleTest")

    // Create GameEstimator and fit model
    val estimator = new MockGameEstimator(sc, createLogger("SimpleTest"))
    estimator
      .set(estimator.trainingTask, TaskType.LINEAR_REGRESSION)
      .set(estimator.coordinateUpdateSequence, Seq(coordinateId))
    val coordinateDescent = new CoordinateDescent(
      estimator.getOrDefault(estimator.coordinateUpdateSequence),
      estimator.getOrDefault(estimator.coordinateDescentIterations),
      validationDataAndEvaluationSuiteOpt = None,
      lockedCoordinates = Set(),
      logger)
    val models: (GameModel, Option[EvaluationResults]) = estimator.train(
      modelConfig,
      trainingDatasets,
      coordinateDescent)
    val model = models._1.getModel(coordinateId).get.asInstanceOf[FixedEffectModel].model

    // Reference values from scikit-learn
    assertEquals(model.coefficients.means(0), 0.3215554473500486, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(model.coefficients.means(1), 0.17904355431985355, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(model.coefficients.means(2), 0.4122241763914806, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }

  /**
   * Combine the trivial dataset with each type of normalization.
   *
   * @return A dataset to train on and the type or normalization to use during training
   */
  @DataProvider
  def normalizationParameters(): Array[Array[Object]] = {
    val labeledPoints = trivialLabeledPoints()(0)(0)

    NormalizationType
      .values
      .map(normalization => Array(labeledPoints, normalization))
      .toArray
  }

  /**
   * In this test, we just verify that we can fit a model with each possible normalization type, without verifying that
   * the results are correct.
   *
   * @note When regularization is not involved, the model should be invariant under renormalization, since we undo
   *       normalization after training.
   */
  @Test(dataProvider = "normalizationParameters")
  def testNormalization(labeledPoints: Seq[LabeledPoint], normalizationType: NormalizationType): Unit =
    sparkTest("testNormalization") {

      val (coordinateId, featureShardId) = ("global", "features")

      val trainingDataSeq = labeledPoints
        .zipWithIndex
        .map { case(point, index) =>

          val newPoint = LabeledPoint(point.label, new DenseVector[Double](point.features.toArray :+ 1.0), 0D, 1D)
          (index.toLong, newPoint)
        }
      val trainingDataRdd = sc.parallelize(trainingDataSeq)
      val fixedEffectDataset = new FixedEffectDataset(trainingDataRdd, featureShardId)
      val trainingDatasets = Map((coordinateId, fixedEffectDataset))

      val fixedEffectOptConfig = FixedEffectOptimizationConfiguration(
        OptimizerConfig(
          optimizerType = OptimizerType.LBFGS,
          maximumIterations = 100,
          tolerance = 1e-11,
          constraintMap = None))
      val modelConfig: GameEstimator.GameOptimizationConfiguration = Map((coordinateId, fixedEffectOptConfig))

      val statisticalSummary = FeatureDataStatistics(trainingDataRdd.values, Some(labeledPoints.head.features.length))
      val normalizationContext = NormalizationContext(normalizationType, statisticalSummary)
      val normalizationContexts = Map((coordinateId, normalizationContext))
      val logger = createLogger("NormalizationTest")

      // Create GameEstimator and fit model
      val estimator = new MockGameEstimator(sc, logger)
      estimator
        .set(estimator.trainingTask, TaskType.LINEAR_REGRESSION)
        .set(estimator.coordinateUpdateSequence, Seq(coordinateId))
        .set(estimator.coordinateNormalizationContexts, normalizationContexts)
      val coordinateDescent = new CoordinateDescent(
        estimator.getOrDefault(estimator.coordinateUpdateSequence),
        estimator.getOrDefault(estimator.coordinateDescentIterations),
        validationDataAndEvaluationSuiteOpt = None,
        lockedCoordinates = Set(),
        logger)
      val models: (GameModel, Option[EvaluationResults]) = estimator.train(
        modelConfig,
        trainingDatasets,
        coordinateDescent)
      val model = models._1.getModel(coordinateId).get.asInstanceOf[FixedEffectModel].model

      // Reference values from scikit-learn
      assertEquals(model.coefficients.means(0), 0.34945501725815586, CommonTestUtils.LOW_PRECISION_TOLERANCE)
      assertEquals(model.coefficients.means(1), 0.26339479490270173, CommonTestUtils.LOW_PRECISION_TOLERANCE)
      assertEquals(model.coefficients.means(2), 0.4366125400310442, CommonTestUtils.LOW_PRECISION_TOLERANCE)
    }

  /**
   * Test that the [[GameEstimator]] can correctly convert a [[DataFrame]] of training data into one or more
   * [[Dataset]]s.
   */
  @Test
  def testPrepareTrainingDatasets(): Unit = sparkTest("testPrepareTrainingDataSetsAndEvaluator") {

    // Load DataFrame
    val data = getData(sparkSession)

    // Create GameTrainingDriver
    val estimator = new MockGameEstimator(sc, createLogger("testPrepareTrainingDatasetsAndEvaluator"))
    estimator
      .set(estimator.trainingTask, TaskType.LINEAR_REGRESSION)
      .set(estimator.coordinateDataConfigurations, coordinateDataConfigurations)
    val featureShards = coordinateDataConfigurations
      .map { case (_, coordinateDataConfig) =>
        coordinateDataConfig.featureShardId
      }
      .toSet
    val trainingDatasets = estimator.prepareTrainingDatasets(data, featureShards, idTagSet)

    assertEquals(trainingDatasets.size, 4)

    // global data
    trainingDatasets("global") match {
      case ds: FixedEffectDataset =>
        assertEquals(ds.labeledPoints.count(), 34810)
        assertEquals(ds.numFeatures, 30085)

      case _ => fail("Wrong dataset type.")
    }

    // per-user data
    trainingDatasets("per-user") match {
      case ds: RandomEffectDataset =>
        assertEquals(ds.activeData.count(), 33110)

        val featureStats = ds.activeData.values.map(_.numActiveFeatures).stats()
        assertEquals(featureStats.count, 33110)
        assertEquals(featureStats.mean, 24.12999093, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.stdev, 0.61119425, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.max, 40.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.min, 24.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)

      case _ => fail("Wrong dataset type.")
    }

    // per-song data
    trainingDatasets("per-song") match {
      case ds: RandomEffectDataset =>
        assertEquals(ds.activeData.count(), 23167)

        val featureStats = ds.activeData.values.map(_.numActiveFeatures).stats()
        assertEquals(featureStats.count, 23167)
        assertEquals(featureStats.mean, 21.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.stdev, 0.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.max, 21.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.min, 21.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)

      case _ => fail("Wrong dataset type.")
    }

    // per-artist data
    trainingDatasets("per-artist") match {
      case ds: RandomEffectDataset =>
        assertEquals(ds.activeData.count(), 4471)

        val featureStats = ds.activeData.values.map(_.numActiveFeatures).stats()
        assertEquals(featureStats.count, 4471)
        assertEquals(featureStats.mean, 3.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.stdev, 0.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.max, 3.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
        assertEquals(featureStats.min, 3.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)

      case _ => fail("Wrong dataset type.")
    }
  }

  /**
   * Provide a list of validation [[EvaluatorType]]s, in varying combinations.
   *
   * @return A list of [[EvaluatorType]] and/or [[MultiEvaluatorType]]
   */
  @DataProvider
  def multipleEvaluatorTypeProvider(): Array[Array[Any]] =
    Array(
      Array(Seq(RMSE, SquaredLoss)),
      Array(Seq(LogisticLoss, AUC, MultiPrecisionAtK(1, "userId"), MultiPrecisionAtK(10, "songId"))),
      Array(Seq(AUC, MultiAUC("userId"), MultiAUC("songId"))),
      Array(Seq(PoissonLoss)))

  /**
   * Test that the [[GameEstimator]] correctly instantiates validation [[Evaluator]]s from a list of [[EvaluatorType]]s.
   *
   * @param evaluatorTypes The list of validation [[EvaluatorType]]s
   */
  @Test(dataProvider = "multipleEvaluatorTypeProvider")
  def testMultipleEvaluators(evaluatorTypes: Seq[EvaluatorType]): Unit =
    sparkTest("multipleEvaluatorsWithFullModel", useKryo = true) {
      val evaluatorCols = MultiEvaluatorType.getMultiEvaluatorIdTags(evaluatorTypes)
      val mockValidationData = getMockDataRDD(evaluatorCols)
      val estimator = new MockGameEstimator(sc, createLogger("taskAndDefaultEvaluatorTypeProvider"))
        .setValidationEvaluators(evaluatorTypes)
      val evaluationSuite = estimator.prepareValidationEvaluators(mockValidationData)
      val evaluationSuiteTypes = evaluationSuite.evaluators.map(_.evaluatorType)

      assertEquals(evaluationSuiteTypes.size, evaluatorTypes.size)
      evaluatorTypes.foreach(evaluationSuiteTypes.contains)
    }

  /**
   * Provide a combination of training task and default validation [[EvaluatorType]] for that task.
   *
   * @return A training task as a [[TaskType]] and an [[EvaluatorType]]
   */
  @DataProvider
  def taskAndDefaultEvaluatorTypeProvider(): Array[Array[Any]] =
    Array(
      Array(TaskType.LINEAR_REGRESSION, RMSE),
      Array(TaskType.LOGISTIC_REGRESSION, AUC),
      Array(TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, AUC),
      Array(TaskType.POISSON_REGRESSION, PoissonLoss))

  /**
   * Test that the [[GameEstimator]] correctly instantiates a default validation [[Evaluator]] for the given training
   * task.
   *
   * @param taskType The training task
   * @param defaultEvaluatorType The [[EvaluatorType]] of the correct default validation [[Evaluator]] for the training
   *                             task
   */
  @Test(dataProvider = "taskAndDefaultEvaluatorTypeProvider")
  def testDefaultEvaluator(
    taskType: TaskType,
    defaultEvaluatorType: EvaluatorType): Unit = sparkTest("taskAndDefaultEvaluatorTypeProvider") {

    val mockValidationData = getMockDataRDD()
    val estimator = new MockGameEstimator(sc, createLogger("taskAndDefaultEvaluatorTypeProvider"))
      .setTrainingTask(taskType)
    val evaluationSuite = estimator.prepareValidationEvaluators(mockValidationData)

    assertEquals(evaluationSuite.primaryEvaluator.evaluatorType.name, defaultEvaluatorType.name)
  }

  /**
   * Create a mock [[GameDatum]] [[RDD]] containing only one point.
   *
   * @param evaluatorCols Custom evaluator columns
   * @return A mocked GAME data [[RDD]]
   */
  def getMockDataRDD(evaluatorCols: Set[String] = Set()): RDD[(UniqueSampleId, GameDatum)] = {
    val id: UniqueSampleId = 1L
    val response = 1D
    val offset = 0D
    val weight = 1D
    val featureShardId: FeatureShardId = "shard1"
    val features: Vector[Double] = VectorUtils.toDenseVector(Array((0, 1D)), 1)
    val additionalIds: Map[String, String] = evaluatorCols.map(id => id -> "").toMap
    val gameDatum = new GameDatum(response, Some(offset), Some(weight), Map(featureShardId -> features), additionalIds)

    sc.parallelize(Seq((id, gameDatum)))
  }

  /**
   * Returns the test case data frame.
   *
   * @return A loaded data frame
   */
  def getData(spark: SparkSession): DataFrame = spark.read.parquet(inputPath)

  /**
   * Creates a test case Photon logger.
   *
   * @param testName Optional name of the test; if provided the logs go to that sub-dir in the tmp dir (tmp dir is per
   *                 thread)
   * @return A created logger
   */
  def createLogger(testName: String = "GenericTest"): PhotonLogger = new PhotonLogger(s"$getTmpDir/$testName", sc)
}

object GameEstimatorIntegTest {

  /**
   * The test dataset here is a subset of the Yahoo! music dataset available on the internet, in [[DataFrame]] form,
   * stored in the parquet format.
   */
  private val inputPath = getClass
    .getClassLoader
    .getResource("GameEstimatorIntegTest/input/yahooMusic/train")
    .getPath

  /**
   * Data configurations for the above test data.
   */
  private val coordinateDataConfigurations = Map(
    "global" -> FixedEffectDataConfiguration("shard1", 2),
    "per-user" -> RandomEffectDataConfiguration("userId", "shard2", 2),
    "per-song" -> RandomEffectDataConfiguration("songId", "shard3", 2),
    "per-artist" -> RandomEffectDataConfiguration("artistId", "shard3", 2, projectorType = RandomProjection(2)))
  private val idTagSet = Set("userId", "songId", "artistId")

  /**
   * Mock [[GameEstimator]] for testing purposes - it exposes protected members so that they can be called by the test
   * cases.
   */
  private class MockGameEstimator(sc: SparkContext, logger: Logger) extends GameEstimator(sc, logger) {

    override def prepareTrainingDatasets(
        data: DataFrame,
        featureShards: Set[FeatureShardId],
        additionalCols: Set[String]): Map[CoordinateId, D forSome { type D <: Dataset[D] }] =
      super.prepareTrainingDatasets(data, featureShards, additionalCols)

    override def prepareValidationDatasetAndEvaluators(
        dataOpt: Option[DataFrame],
        featureShards: Set[FeatureShardId],
        additionalCols: Set[String]): Option[(RDD[(UniqueSampleId, GameDatum)], EvaluationSuite)] =
      super.prepareValidationDatasetAndEvaluators(dataOpt, featureShards, additionalCols)

    override def prepareValidationEvaluators(gameDataset: RDD[(UniqueSampleId, GameDatum)]): EvaluationSuite =
      super.prepareValidationEvaluators(gameDataset)

    override def prepareNormalizationContextWrappers(datasets: Map[CoordinateId, D forSome { type D <: Dataset[D] } ])
      : Option[Map[CoordinateId, NormalizationContextWrapper]] =

      super.prepareNormalizationContextWrappers(datasets)

    override def train(
        configuration: GameEstimator.GameOptimizationConfiguration,
        trainingDatasets: Map[CoordinateId, D forSome { type D <: Dataset[D] }],
        coordinateDescent: CoordinateDescent,
        normalizationContextWrappersOpt: Option[Map[CoordinateId, NormalizationContextWrapper]] = None,
        prevGameModelOpt: Option[GameModel] = None,
        flag: Boolean = false): (GameModel, Option[EvaluationResults]) =
      super.train(configuration, trainingDatasets, coordinateDescent, normalizationContextWrappersOpt, prevGameModelOpt)
  }
}
