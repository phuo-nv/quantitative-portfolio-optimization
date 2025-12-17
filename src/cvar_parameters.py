# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class CvarParameters:
    """
    User‑tunable parameters and constraint limits for CVaR optimization.

    Most parameters are scalars. Weight bounds ``w_min`` / ``w_max`` can be:
    - numpy arrays (length n_assets) for per-asset bounds
    - dict mapping asset names to bounds
    - float for uniform bounds across all assets
    - None for no bounds

    Optional constraints (T_tar, cvar_limit, cardinality) default to None when
    not specified.
    """

    # Weight / cash bounds
    w_min: Union[np.ndarray, dict, float] = 1.0  # Lower bound for each risky weight
    w_max: Union[np.ndarray, dict, float] = 0.0  # Upper bound for each risky weight
    c_min: float = 0  # Lower bound for cash allocation
    c_max: float = 1  # Upper bound for cash allocation
    # Risk model Parameters
    risk_aversion: float = 1  # λ – penalty applied to CVaR inside objective
    confidence: float = 0.95  # α in CVaR_α (e.g. 0.95 -> 95 % CVaR)
    # Soft / hard constraint targets
    L_tar: float = 1.6  # Leverage constraint (Σ|wᵢ|)
    T_tar: Optional[float] = None  # Turnover constraint
    cvar_limit: Optional[float] = None  # Hard CVaR limit (None means "no hard limit")
    cardinality: Optional[int] = None  # number of assets to be selected
    group_constraints: Optional[list[dict]] = None
    # Group constraints:
    # [{'group_name': group_name,
    #   'tickers': tickers
    #   'weight_bounds': {'w_min': w_min, 'w_max': w_max}}]

    def __post_init__(self) -> None:
        """Validate initial parameter values."""
        self._validate_c_min(self.c_min)
        self._validate_c_max(self.c_max)
        self._validate_risk_aversion(self.risk_aversion)
        self._validate_confidence(self.confidence)
        if self.cardinality is not None:
            self._validate_cardinality(self.cardinality)

    # --- Validation helpers ---

    @staticmethod
    def _validate_c_min(value: float) -> None:
        if value < 0:
            raise ValueError("Cash lower bound (c_min) must be non-negative.")

    @staticmethod
    def _validate_c_max(value: float) -> None:
        if not (0 <= value <= 1):
            raise ValueError("Cash upper bound (c_max) must be in [0, 1].")

    @staticmethod
    def _validate_risk_aversion(value: float) -> None:
        if value < 0:
            raise ValueError("Risk aversion must be non-negative.")

    @staticmethod
    def _validate_confidence(value: float) -> None:
        if not (0 < value <= 1):
            raise ValueError(
                "Confidence level must be in (0, 1], e.g. 0.95 for 95% CVaR."
            )

    @staticmethod
    def _validate_cardinality(value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Cardinality must be a positive integer.")

    def update_w_min(self, new_w_min: Union[np.ndarray, dict, float]):
        self.w_min = new_w_min

    def update_w_max(self, new_w_max: Union[np.ndarray, dict, float]):
        if new_w_max <= 1:
            self.w_max = new_w_max
        else:
            raise ValueError("Invalid upper bound for weights!")

    def update_c_min(self, new_c_min: float):
        self._validate_c_min(new_c_min)
        self.c_min = new_c_min

    def update_c_max(self, new_c_max: float):
        self._validate_c_max(new_c_max)
        self.c_max = new_c_max

    def update_z_min(self, new_c_min: float):
        self.z_min = new_c_min

    def update_z_max(self, new_z_max: float):
        self.z_max = new_z_max

    def update_T_tar(self, new_T_tar: float):
        self.T_tar = new_T_tar

    def update_L_tar(self, new_L_tar: float):
        self.L_tar = new_L_tar

    def update_cvar_limit(self, new_cvar_limit: float):
        self.cvar_limit = new_cvar_limit

    def update_cardinality(self, new_cardinality: int):
        if new_cardinality is not None:
            self._validate_cardinality(new_cardinality)
        self.cardinality = new_cardinality

    def update_risk_aversion(self, new_risk_aversion: float):
        self._validate_risk_aversion(new_risk_aversion)
        self.risk_aversion = new_risk_aversion

    def update_confidence(self, new_confidence: float):
        self._validate_confidence(new_confidence)
        self.confidence = new_confidence
