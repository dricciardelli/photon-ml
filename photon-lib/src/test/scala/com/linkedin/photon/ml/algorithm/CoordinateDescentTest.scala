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

import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.{CoordinateId, UniqueSampleId}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.evaluation.{EvaluationResults, EvaluationSuite, EvaluatorType}
import com.linkedin.photon.ml.model.{DatumScoringModel, GameModel}
import com.linkedin.photon.ml.optimization.OptimizationTracker
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * Unit tests for [[CoordinateDescent]].
 */
class CoordinateDescentTest {

  import CoordinateDescentTest._

  abstract class MockDataset extends Dataset[MockDataset] {}
  type CoordinateType = Coordinate[MockDataset]

  @DataProvider
  def invalidInput: Array[Array[Any]] = {

    // Mock parameters
    val goodIter = 1
    val coordinateId1 = "someCoordinate"
    val coordinateId2 = "someOtherCoordinate"
    val goodUpdateSequence = Seq(coordinateId1)
    val badUpdateSequence = Seq(coordinateId1, coordinateId1)
    val goodLockedCoordinates = Set(coordinateId1)
    val badLockedCoordinates = Set(coordinateId2)

    Array(
      // Repeated coordinates in the update sequence
      Array(badUpdateSequence, goodIter, goodLockedCoordinates),
      // 0 iterations
      Array(goodUpdateSequence, 0, goodLockedCoordinates),
      // Negative iterations
      Array(goodUpdateSequence, -1, goodLockedCoordinates),
      // Locked coordinates missing from update sequence
      Array(goodUpdateSequence, goodIter, badLockedCoordinates))
  }

  /**
   * Test [[CoordinateDescent]] creation with invalid input.
   *
   * @param updateSequence The order in which to update coordinates
   * @param descentIterations Number of coordinate descent iterations (updates to each coordinate in order)
   * @param lockedCoordinates Set of locked coordinates within the initial model for performing partial retraining
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testCheckInvariants(
      updateSequence: Seq[CoordinateId],
      descentIterations: Int,
      lockedCoordinates: Set[CoordinateId]): Unit =
    new CoordinateDescent(
      updateSequence,
      descentIterations,
      None,
      lockedCoordinates,
      MOCK_LOGGER)

  @DataProvider
  def invalidRunInput: Array[Array[Any]] = {

    // Mocks
    val goodGameModel = mock(classOf[GameModel])
    val badGameModel = mock(classOf[GameModel])
    val datumScoringModel = mock(classOf[DatumScoringModel])
    val coordinate = mock(classOf[CoordinateType])

    // Mock parameters
    val updateSequence = Seq("a", "b", "c", "d")
    val lockedCoordinates = Set("a", "c")
    val descentIterations = 1
    val goodCoordinates = Map("a" -> coordinate, "b" -> coordinate,"c" -> coordinate, "d" -> coordinate)
    val badCoordinates = Map("a" -> coordinate, "b" -> coordinate, "d" -> coordinate)

    doReturn(Some(datumScoringModel)).when(goodGameModel).getModel("a")
    doReturn(Some(datumScoringModel)).when(goodGameModel).getModel("b")
    doReturn(Some(datumScoringModel)).when(goodGameModel).getModel("c")
    doReturn(Some(datumScoringModel)).when(goodGameModel).getModel("d")
    doReturn(Some(datumScoringModel)).when(badGameModel).getModel("a")
    doReturn(Some(datumScoringModel)).when(badGameModel).getModel("b")
    doReturn(Some(datumScoringModel)).when(badGameModel).getModel("c")
    doReturn(None).when(badGameModel).getModel("d")

    val coordinateDescent = new CoordinateDescent(
      updateSequence,
      descentIterations,
      validationDataAndEvaluationSuiteOpt = None,
      lockedCoordinates,
      MOCK_LOGGER)

    Array(
      // Coordinates to train are missing from the coordinates map
      Array(coordinateDescent, badCoordinates, goodGameModel),
      // Update sequence coordinates missing from the gameModel
      Array(coordinateDescent, goodCoordinates, badGameModel))
  }

  /**
   * Test attempts to run coordinate descent with invalid input.
   *
   * @param coordinateDescent A pre-built [[CoordinateDescent]] object to attempt to run
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   * @param gameModel The initial GAME model to use as a starting point
   */
  @Test(dataProvider = "invalidRunInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRun(
      coordinateDescent: CoordinateDescent,
      coordinates: Map[CoordinateId, CoordinateType],
      gameModel: GameModel): Unit = coordinateDescent.run(coordinates, gameModel)

  @DataProvider
  def coordinateCountProvider(): Array[Array[Integer]] = (1 to 5).map(x => Array(Int.box(x))).toArray

  /**
   * Tests for [[CoordinateDescent]] without validation.
   *
   * @param coordinateCount Number of coordinates to generate for test
   */
  @Test(dataProvider = "coordinateCountProvider")
  def testRun(coordinateCount: Int): Unit = {

    val numIterations = 10

    // Create Coordinate and model mocks
    val coordinateIds = (0 until coordinateCount).map("coordinate" + _)
    val coordinatesAndModels: Seq[(CoordinateId, CoordinateType, DatumScoringModel)] =
      coordinateIds.map { coordinateId =>
        (coordinateId, mock(classOf[CoordinateType]), mock(classOf[DatumScoringModel]))
      }
    val coordinates = coordinatesAndModels
      .map { case (coordinateId, coordinate, _) =>
        (coordinateId, coordinate)
      }
      .toMap

    // Other mocks
    val gameModel = mock(classOf[GameModel])
    val tracker = mock(classOf[OptimizationTracker])
    val score = mock(classOf[CoordinateDataScores])

    // Scores mock setup
    when(score + score).thenReturn(score)
    when(score - score).thenReturn(score)
    doReturn(score).when(score).setName(Matchers.any(classOf[String]))
    doReturn(score).when(score).persistRDD(Matchers.any())

    // Per-coordinate mock setup
    coordinatesAndModels.foreach { case (coordinateId, coordinate, model) =>

      // Coordinate mock setup
      doReturn(score).when(coordinate).score(model)
      doReturn((model, Some(tracker))).when(coordinate).updateModel(model)
      doReturn((model, Some(tracker))).when(coordinate).updateModel(model, score)

      // GameModel mock setup
      doReturn(Map[CoordinateId, DatumScoringModel]()).when(gameModel).toMap
      doReturn(gameModel).when(gameModel).updateModel(coordinateId, model)
      doReturn(Some(model)).when(gameModel).getModel(coordinateId)
    }

    // Run coordinate descent - None = no validation
    val coordinateDescent = new CoordinateDescent(
      coordinateIds,
      numIterations,
      validationDataAndEvaluationSuiteOpt = None,
      lockedCoordinates = Set(),
      MOCK_LOGGER)
    coordinateDescent.run(coordinates, gameModel)

    // Verify the calls to updateModel
    coordinatesAndModels.foreach { case (_, coordinate, model) =>
      if (coordinates.size == 1) {
        verify(coordinate, times(numIterations)).updateModel(model)
      } else {
        verify(coordinate, times(numIterations)).updateModel(model, score)
      }
    }
  }

  /**
   * Test [[CoordinateDescent]] with validation.
   */
  @Test
  def testBestModel(): Unit = {

    val iterationCount = 3

    val coordinate = mock(classOf[CoordinateType])
    val coordinateIds = Seq("Coordinate0")
    val coordinates: Map[CoordinateId, CoordinateType] = Map(coordinateIds.head -> coordinate)

    val tracker = mock(classOf[OptimizationTracker])
    val (score, validationScore) = (mock(classOf[CoordinateDataScores]), mock(classOf[CoordinateDataScores]))
    val coordinateModel = mock(classOf[DatumScoringModel])
    val modelScores = mock(classOf[RDD[(UniqueSampleId, Double)]])
    val validationData = mock(classOf[RDD[(UniqueSampleId, GameDatum)]])

    val mockEvaluatorType1 = mock(classOf[EvaluatorType])
    val mockEvaluatorType2 = mock(classOf[EvaluatorType])
    val evaluationResults = EvaluationResults(
      Map(mockEvaluatorType1 -> 0D, mockEvaluatorType2 -> 0D),
      mockEvaluatorType1)
    val evaluationSuite = mock(classOf[EvaluationSuite])

    when(score + score).thenReturn(score)
    when(score - score).thenReturn(score)
    when(score.setName(Matchers.any())).thenReturn(score)
    when(score.persistRDD(Matchers.any())).thenReturn(score)

    when(validationScore + validationScore).thenReturn(validationScore)
    when(validationScore - validationScore).thenReturn(validationScore)
    when(validationScore.scoresRdd).thenReturn(modelScores)
    when(validationScore.setName(Matchers.any())).thenReturn(validationScore)
    when(validationScore.persistRDD(Matchers.any())).thenReturn(validationScore)
    when(validationScore.materialize()).thenReturn(validationScore)

    when(evaluationSuite.evaluate(modelScores)).thenReturn(evaluationResults)
    // The evaluators are called iterationCount - 1 times, since on the first iteration
    // there is nothing to compare the model against
    when(mockEvaluatorType1.betterThan(Matchers.any(), Matchers.any()))
      .thenReturn(true)
      .thenReturn(false)
    when(mockEvaluatorType2.betterThan(Matchers.any(), Matchers.any()))
      .thenReturn(false)
      .thenReturn(false)

    // The very first GAME model will give rise to GAME model #1 via update,
    // and GAME model #1 will be the first one to go through best model selection
    val gameModels = (0 to iterationCount + 1).map { _ => mock(classOf[GameModel]) }
    (0 until iterationCount).map { i =>
      when(gameModels(i).toMap).thenReturn(Map[CoordinateId, DatumScoringModel]())
      when(gameModels(i).getModel(Matchers.any())).thenReturn(Some(coordinateModel))
      when(gameModels(i).updateModel(Matchers.any(), Matchers.any())).thenReturn(gameModels(i + 1))
    }

    when(coordinateModel.scoreForCoordinateDescent(Matchers.any())).thenReturn(validationScore)
    when(coordinate.score(Matchers.any())).thenReturn(score)
    when(coordinate.updateModel(coordinateModel)).thenReturn((coordinateModel, Some(tracker)))
    when(coordinate.updateModel(coordinateModel, score)).thenReturn((coordinateModel, Some(tracker)))

    // Run coordinate descent
    val validationDataAndEvaluationSuite = (validationData, evaluationSuite)
    val coordinateDescent = new CoordinateDescent(
      coordinateIds,
      iterationCount,
      Some(validationDataAndEvaluationSuite),
      lockedCoordinates = Set(),
      MOCK_LOGGER)
    val (returnedModel, _) = coordinateDescent.run(coordinates, gameModels.head)

    assertEquals(returnedModel.hashCode(), gameModels(2).hashCode())

    // Verify the calls to updateModel
    verify(coordinate, times(iterationCount)).updateModel(coordinateModel)

    // Verify the calls to the validation evaluator set
    verify(validationDataAndEvaluationSuite._2, times(iterationCount)).evaluate(modelScores)
  }
}

object CoordinateDescentTest {

  val MOCK_LOGGER: PhotonLogger = mock(classOf[PhotonLogger])
}
