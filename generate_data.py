"""
generate_data.py
Generates synthetic telecom-like time series data with injected anomalies.
Mimics signal success rate metrics (e.g. S1 Signalling Connection Establishment Success Rate)
No real telecom data is used.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_telecom_timeseries(
    n_points=2000,
    start_date="2023-01-01",
    freq_minutes=15,
    noise_std=0.02,
    anomaly_configs=None,
    seed=42
):
    """
    Generate synthetic telecom signal success rate time series.
    
    Parameters:
        n_points: number of time steps
        start_date: start of the time series
        freq_minutes: frequency of measurements in minutes
        noise_std: standard deviation of Gaussian noise
        anomaly_configs: list of dicts with anomaly definitions
        seed: random seed for reproducibility
    
    Returns:
        pd.DataFrame with columns [Timestamp, SignalSuccessRate, is_anomaly]
    """
    np.random.seed(seed)
    
    # Generate timestamps
    start = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start + timedelta(minutes=freq_minutes * i) for i in range(n_points)]
    
    # Base signal: ~0.95-1.0 with daily and weekly seasonality
    t = np.arange(n_points)
    
    # Daily pattern: slight dip at night
    daily_cycle = 0.02 * np.sin(2 * np.pi * t / (24 * 60 / freq_minutes))
    
    # Weekly pattern: lower on weekends
    weekly_cycle = 0.01 * np.sin(2 * np.pi * t / (7 * 24 * 60 / freq_minutes))
    
    # Base signal
    signal = 0.97 + daily_cycle + weekly_cycle
    
    # Add Gaussian noise
    signal += np.random.normal(0, noise_std, n_points)
    
    # Clip to valid range
    signal = np.clip(signal, 0.0, 1.0)
    
    # Anomaly labels
    is_anomaly = np.zeros(n_points, dtype=int)
    
    # Default anomaly configs if none provided
    if anomaly_configs is None:
        anomaly_configs = [
            # Sudden drop - hardware failure simulation
            {"start": 300, "duration": 20, "type": "drop", "magnitude": 0.5},
            # Gradual drift - network degradation simulation  
            {"start": 700, "duration": 50, "type": "drift", "magnitude": 0.3},
            # Spike - interference event simulation
            {"start": 1100, "duration": 5, "type": "spike", "magnitude": 0.4},
            # Sustained degradation - congestion simulation
            {"start": 1500, "duration": 80, "type": "drop", "magnitude": 0.25},
            # Brief dropout
            {"start": 1800, "duration": 10, "type": "drop", "magnitude": 0.6},
        ]
    
    # Inject anomalies
    for config in anomaly_configs:
        start_idx = config["start"]
        end_idx = min(start_idx + config["duration"], n_points)
        anomaly_type = config["type"]
        magnitude = config["magnitude"]
        
        if anomaly_type == "drop":
            signal[start_idx:end_idx] -= magnitude
            signal[start_idx:end_idx] = np.clip(signal[start_idx:end_idx], 0.0, 1.0)
            
        elif anomaly_type == "drift":
            # Gradual linear drift downward
            drift = np.linspace(0, magnitude, end_idx - start_idx)
            signal[start_idx:end_idx] -= drift
            signal[start_idx:end_idx] = np.clip(signal[start_idx:end_idx], 0.0, 1.0)
            
        elif anomaly_type == "spike":
            # Brief spike upward or downward
            signal[start_idx:end_idx] += magnitude * np.random.choice([-1, 1])
            signal[start_idx:end_idx] = np.clip(signal[start_idx:end_idx], 0.0, 1.0)
        
        is_anomaly[start_idx:end_idx] = 1
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "SignalSuccessRate": np.round(signal, 4),
        "is_anomaly": is_anomaly
    })
    
    return df


def generate_multi_metric_timeseries(n_points=2000, seed=42):
    """
    Generate multiple correlated telecom metrics.
    Mimics real network monitoring scenarios.
    """
    np.random.seed(seed)
    
    # Generate base signal
    df = generate_telecom_timeseries(n_points=n_points, seed=seed)
    
    # Correlated metrics
    # Throughput: correlated with signal success rate
    df["Throughput_Mbps"] = (
        50 + 30 * df["SignalSuccessRate"] 
        + np.random.normal(0, 2, n_points)
    )
    
    # Latency: inversely correlated with signal quality
    df["Latency_ms"] = (
        20 + 40 * (1 - df["SignalSuccessRate"]) 
        + np.random.normal(0, 1, n_points)
    )
    
    # Packet loss: spikes when signal drops
    df["PacketLoss_pct"] = (
        np.clip(
            0.5 * (1 - df["SignalSuccessRate"]) * 100 
            + np.random.exponential(0.1, n_points),
            0, 100
        )
    )
    
    return df


if __name__ == "__main__":
    print("Generating synthetic telecom time series data...")
    
    # Single metric
    df_single = generate_telecom_timeseries(n_points=2000)
    df_single.to_csv("data/synthetic_telecom_single.csv", index=False)
    print(f"Single metric: {len(df_single)} rows, {df_single['is_anomaly'].sum()} anomaly points")
    
    # Multi metric
    df_multi = generate_multi_metric_timeseries(n_points=2000)
    df_multi.to_csv("data/synthetic_telecom_multi.csv", index=False)
    print(f"Multi metric: {len(df_multi)} rows, {df_multi['is_anomaly'].sum()} anomaly points")
    
    print("\nSample data:")
    print(df_single.head(10))
    print("\nAnomaly distribution:")
    print(df_single['is_anomaly'].value_counts())
