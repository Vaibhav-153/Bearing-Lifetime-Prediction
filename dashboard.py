"""Streamlit dashboard for the bearing RUL prediction API."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "https://bearing-remaining-lifetime-prediction.onrender.com/predict"
API_URL = os.getenv("BEARING_API_URL", DEFAULT_API_URL)
WINDOW_SIZE = 15
REQUEST_TIMEOUT_SECONDS = 45


def read_signal(uploaded_file) -> list[float]:
    """Read the first vibration channel from one IMS-style text file."""
    frame = pd.read_csv(uploaded_file, sep="\t", header=None)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("File contains no signal data.")

    signal = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    if signal.size == 0:
        raise ValueError("Signal is empty.")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")

    return signal.tolist()


def parse_error_response(response: requests.Response) -> str:
    """Return a useful API error message even when the body is not JSON."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Unknown API error"

    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)


st.set_page_config(page_title="Bearing RUL Predictor", layout="wide")
st.title("Bearing Remaining Useful Life Predictor")
st.write(
    "Upload 15 chronological vibration files. The dashboard reads the first signal "
    "column from each file and sends the sequence to the prediction API."
)

uploaded_files = st.file_uploader(
    f"Upload exactly {WINDOW_SIZE} sequential tab-separated files.",
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) != WINDOW_SIZE:
        st.warning(
            f"Upload exactly {WINDOW_SIZE} files. Currently selected: {len(uploaded_files)}."
        )
    else:
        ordered_files = sorted(uploaded_files, key=lambda file: file.name)
        st.success(f"{len(ordered_files)} files are ready. Files will be processed by name.")

        if st.button("Predict RUL", type="primary"):
            with st.spinner("Reading files and requesting a prediction..."):
                try:
                    signals = [read_signal(file) for file in ordered_files]
                    response = requests.post(
                        API_URL,
                        json={"signals": signals},
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                except (ValueError, pd.errors.ParserError) as exc:
                    st.error(f"Could not read the uploaded files: {exc}")
                except requests.RequestException as exc:
                    st.error(f"Could not reach the prediction API: {exc}")
                else:
                    if response.ok:
                        result = response.json()
                        predicted_cycles = result.get("predicted_rul")
                        if predicted_cycles is None:
                            st.error("The API response did not contain 'predicted_rul'.")
                            st.json(result)
                        else:
                            predicted_hours = (float(predicted_cycles) * 10.0) / 60.0
                            st.metric(
                                "Predicted Remaining Useful Life",
                                f"{predicted_hours:.1f} hours",
                            )
                            st.caption(f"Raw model output: {float(predicted_cycles):.2f} cycles")
                    else:
                        st.error(
                            f"API request failed ({response.status_code}): "
                            f"{parse_error_response(response)}"
                        )
