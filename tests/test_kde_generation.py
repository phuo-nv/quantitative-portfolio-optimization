# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cufolio.cvar_utils import generate_samples_kde


class TestGenerateSamplesKDE:
    """Tests for generate_samples_kde function."""

    def test_column_order_preserved_with_constant_columns(self):
        """Verify that output columns match input column order when constant columns exist."""
        # Create test data with some constant columns interspersed
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            "asset_A": np.random.randn(n_samples) * 0.02,
            "constant_1": np.ones(n_samples) * 0.5,  # constant column
            "asset_B": np.random.randn(n_samples) * 0.03,
            "constant_2": np.zeros(n_samples),  # constant column
            "asset_C": np.random.randn(n_samples) * 0.01,
        })
        
        original_columns = data.columns.tolist()
        
        kde_settings = {
            "device": "CPU",
            "bandwidth": 0.1,
            "kernel": "gaussian",
        }
        
        result = generate_samples_kde(
            num_scen=50,
            returns_data=data,
            kde_settings=kde_settings,
            verbose=False,
        )
        
        # Result should have same number of columns as input
        assert result.shape[1] == len(original_columns), (
            f"Expected {len(original_columns)} columns, got {result.shape[1]}"
        )
        
        # Verify constant columns have their original constant values
        result_df = pd.DataFrame(result, columns=original_columns)
        assert np.allclose(result_df["constant_1"], 0.5), (
            "constant_1 should have value 0.5"
        )
        assert np.allclose(result_df["constant_2"], 0.0), (
            "constant_2 should have value 0.0"
        )

        # Verify means of generated samples are close to original data means
        # This ensures column order is preserved (swapped columns would have wrong means)
        original_means = data.mean()
        result_means = result_df.mean()
        for col in original_columns:
            assert np.isclose(result_means[col], original_means[col], atol=0.05), (
                f"Mean of {col} differs: expected ~{original_means[col]:.4f}, "
                f"got {result_means[col]:.4f}"
            )

    def test_column_order_preserved_no_constant_columns(self):
        """Verify column order is preserved when there are no constant columns."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            "asset_A": np.random.randn(n_samples) * 0.02,
            "asset_B": np.random.randn(n_samples) * 0.03,
            "asset_C": np.random.randn(n_samples) * 0.01,
        })
        
        original_columns = data.columns.tolist()
        
        kde_settings = {
            "device": "CPU",
            "bandwidth": 0.1,
            "kernel": "gaussian",
        }
        
        result = generate_samples_kde(
            num_scen=50,
            returns_data=data,
            kde_settings=kde_settings,
            verbose=False,
        )
        
        assert result.shape[1] == len(original_columns)

        # Verify means of generated samples are close to original data means
        result_df = pd.DataFrame(result, columns=original_columns)
        original_means = data.mean()
        result_means = result_df.mean()
        for col in original_columns:
            assert np.isclose(result_means[col], original_means[col], atol=0.05), (
                f"Mean of {col} differs: expected ~{original_means[col]:.4f}, "
                f"got {result_means[col]:.4f}"
            )

    def test_column_order_preserved_all_constant_except_one(self):
        """Edge case: only one non-constant column."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            "constant_1": np.ones(n_samples) * 0.1,
            "asset_A": np.random.randn(n_samples) * 0.02,
            "constant_2": np.ones(n_samples) * 0.2,
            "constant_3": np.ones(n_samples) * 0.3,
        })
        
        original_columns = data.columns.tolist()
        
        kde_settings = {
            "device": "CPU",
            "bandwidth": 0.1,
            "kernel": "gaussian",
        }
        
        result = generate_samples_kde(
            num_scen=50,
            returns_data=data,
            kde_settings=kde_settings,
            verbose=False,
        )
        
        assert result.shape[1] == len(original_columns)
        
        result_df = pd.DataFrame(result, columns=original_columns)
        assert np.allclose(result_df["constant_1"], 0.1)
        assert np.allclose(result_df["constant_2"], 0.2)
        assert np.allclose(result_df["constant_3"], 0.3)

        # Verify means of generated samples are close to original data means
        original_means = data.mean()
        result_means = result_df.mean()
        for col in original_columns:
            assert np.isclose(result_means[col], original_means[col], atol=0.05), (
                f"Mean of {col} differs: expected ~{original_means[col]:.4f}, "
                f"got {result_means[col]:.4f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

