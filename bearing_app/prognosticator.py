# prognosticator.py
# ==============================================================================
# This file contains the core logic for our RUL prediction.
# It is designed as a reusable class that encapsulates the feature extraction
# and model inference steps. This is our "Prediction Engine".
# ==============================================================================

import xgboost as xgb
import numpy as np
import pywt
import os
from typing import List

class BearingPrognosticator:
    """
    A class to predict the Remaining Useful Life (RUL) of a bearing
    using a pre-trained model and a wavelet-based feature extraction method.
    """

    def __init__(self, model_path: str, config: dict):
        """
        Initializes the BearingPrognosticator.

        Args:
            model_path (str): The file path to the pre-trained XGBoost model.
            config (dict): A dictionary containing 'optimal_beta', 'optimal_alpha',
                           and 'window_size'.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load the pre-trained XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        # Load parameters from the configuration dictionary
        self.beta = config['optimal_beta']
        self.alpha = config['optimal_alpha']
        self.window_size = config['window_size']
        
        self.wavelet_name = 'morl' # Using the Morlet wavelet

        print("BearingPrognosticator initialized successfully.")
        print(f"  - Model loaded from: {model_path}")
        print(f"  - Wavelet params: β={self.beta}, α={self.alpha}")
        print(f"  - Prediction window size: {self.window_size}")


    def _calculate_hi(self, raw_signal: np.ndarray) -> float:
        """
        Calculates a Health Indicator (HI) from a raw vibration signal.
        This private method applies the tuned wavelet filter and computes the RMS.

        Args:
            raw_signal (np.ndarray): A 1-D array of raw signal data points.

        Returns:
            float: The calculated Health Indicator (RMS of the filtered signal).
        """
        # Apply the continuous wavelet transform using the pre-defined optimal scale (alpha)
        coeffs, _ = pywt.cwt(raw_signal, scales=[self.alpha], wavelet=self.wavelet_name)
        
        # We use the real part of the coefficients which represents the filtered signal
        filtered_signal = coeffs.real.flatten()
        
        # Calculate Root Mean Square (RMS) as the Health Indicator
        rms = np.sqrt(np.mean(filtered_signal**2))
        return rms

    def predict_rul(self, raw_signal_sequence: List[list]) -> float:
        """
        Predicts the RUL based on a sequence of historical raw signals.

        Args:
            raw_signal_sequence (List[list]): A list of raw signals, where each
                inner list is a signal snapshot. The length of the outer list
                must equal `self.window_size`.

        Returns:
            float: The predicted Remaining Useful Life (RUL).
        """
        # --- Input Validation ---
        if len(raw_signal_sequence) != self.window_size:
            raise ValueError(
                f"Input sequence length ({len(raw_signal_sequence)}) does not match "
                f"the required window size ({self.window_size})."
            )

        # --- Feature Extraction ---
        # 1. Calculate the Health Indicator for each signal in the input sequence
        #    Convert inner lists to numpy arrays for calculation.
        health_indicators = [self._calculate_hi(np.array(signal)) for signal in raw_signal_sequence]
        
        # --- Model Inference ---
        # 2. Format the HI scores into the 2D shape expected by the XGBoost model
        #    The model was trained on (n_samples, n_features), so we need (1, window_size)
        feature_vector = np.array(health_indicators).reshape(1, -1)
        
        # 3. Make the prediction using the loaded model
        predicted_rul = self.model.predict(feature_vector)
        
        # The output of .predict() is an array, so we return the single scalar value
        return float(predicted_rul[0])
