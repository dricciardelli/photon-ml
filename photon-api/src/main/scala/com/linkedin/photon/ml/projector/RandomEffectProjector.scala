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

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

// TODO - Documentation

/**
 * A trait that performs two types of projections:
 * <ul>
 * <li>
 *   Project the random effect dataset from the original space to the projected space, usually as a pre-processing
 *   step before the model training phase.
 * </li>
 * <li>
 *   Project the model coefficients from the projected space back to the original space after training the model,
 *   before scoring a dataset in the original space.
 * </li>
 * </ul>
 */
protected[ml] trait RandomEffectProjector {

  /**
   * Project the active dataset from the original space to the projected space.
   *
   * @param activeData
   * @return The same active dataset in the projected space
   */
  def projectActiveData(activeData: ActiveData): ActiveData

  /**
   * Project the passive dataset from the original space to the projected space.
   *
   * @param passiveData
   * @param passiveDataIds
   * @return The same passive dataset in the projected space
   */
  def projectPassiveData(passiveData: PassiveData, passiveDataIds: Broadcast[Set[REId]]): PassiveData

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the projected space back to the original
   * space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   */
  def projectCoefficientsRDD(modelsRDD: RDD[(REId, GeneralizedLinearModel)]): RDD[(REId, GeneralizedLinearModel)]

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the original space to the projected space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   */
  def transformCoefficientsRDD(modelsRDD: RDD[(REId, GeneralizedLinearModel)]): RDD[(REId, GeneralizedLinearModel)]

}
