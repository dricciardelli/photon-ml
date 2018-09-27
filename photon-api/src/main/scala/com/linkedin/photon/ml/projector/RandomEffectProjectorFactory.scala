/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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

import scala.collection.mutable

import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
import com.linkedin.photon.ml.util.VectorUtils

// TODO - Documentation

/**
 *
 */
object RandomEffectProjectorFactory {

  /**
   * Builds a [[RandomEffectProjector]] object of the appropriate type.
   *
   * @param projectorType The type of the projector
   * @param activeData
   * @param passiveDataOpt
   * @return A new [[RandomEffectProjector]]
   */
  protected[ml] def build(
      projectorType: ProjectorType,
      activeData: ActiveData,
      passiveDataOpt: Option[PassiveData]): RandomEffectProjector = {

    val originalSpaceDimension = activeData.take(1).head._2.numFeatures

    projectorType match {

      case IndexMapProjection =>
        // Collect active indices for the active data set
        val activeDataIndices = activeData.mapValues { ds =>
          ds
            .dataPoints
            .foldLeft(mutable.Set[Int]()){ case (indices, (_, labeledPoint)) =>
              indices ++ VectorUtils.getActiveIndices(labeledPoint.features)
            }
            .toSet
        }

        // Collect active indices for the passive data set
        val passiveDataIndicesOption = passiveDataOpt.map { passiveData =>
          passiveData.map { case (_, (reId, labeledPoint)) =>
            (reId, VectorUtils.getActiveIndices(labeledPoint.features))
          }
        }

        // Union them, and fold the results into (reId, indices) tuples
        val activeIndices = passiveDataIndicesOption
          .map { passiveDataIndices =>
            activeDataIndices
              .union(passiveDataIndices)
              .foldByKey(Set.empty[Int])(_ ++ _)
          }
          .getOrElse(activeIndices)

        IndexMapProjectorRDD(activeIndices, originalSpaceDimension)

      case RandomProjection(projectedSpaceDimension) =>
        ProjectionMatrixBroadcast(originalSpaceDimension, projectedSpaceDimension, isKeepingInterceptTerm = true)

      case IdentityProjection =>
        throw new UnsupportedOperationException(
          s"Projection type '$projectorType' for random effect data set does not require an explicit projector object")

      case _ =>
        throw new UnsupportedOperationException(
          s"Projection type '$projectorType' for random effect data set is not supported")
    }
  }
}
