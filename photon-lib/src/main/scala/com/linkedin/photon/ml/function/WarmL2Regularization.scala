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
package com.linkedin.photon.ml.function

import breeze.linalg.Vector

/**
 *
 */
trait WarmL2Regularization extends L2Regularization {

  protected var priorCoefficients: Vector[Double] = _

  /**
   * Compute the L2 regularization value for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The L2 regularization value
   */
  override protected def l2RegValue(coefficients: Vector[Double]): Double = {

    val regularizedCoefficients = coefficients - priorCoefficients

    l2RegWeight * regularizedCoefficients.dot(regularizedCoefficients) / 2
  }
}

/**
 *
 */
trait WarmL2RegularizationDiff extends L2RegularizationDiff with WarmL2Regularization {

  /**
   * Compute the gradient of the L2 regularization term for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the L2 regularization term
   */
  override protected def l2RegGradient(coefficients: Vector[Double]): Vector[Double] =
    (coefficients - priorCoefficients) * l2RegWeight
}

trait WarmL2RegularizationTwiceDiff extends L2RegularizationTwiceDiff with WarmL2RegularizationDiff
