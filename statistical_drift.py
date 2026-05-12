"""
statistical_drift.py

Hybrid drift and anomaly detection combining:
- Statistical tests (CUSUM, Z-score, IQR)
- Isolation Forest baseline
- HTM-inspired anomaly likelihood scoring

This implements the statistical layer from the paper:
"A Hybrid Framework for Real-Time Data Drift and Anomaly Identification
Using Hierarchical Temporal Memory and Statistical Tests"
Bandyopadhyay S., Bose J., Roychowdhury S.
IJMEMS 2025. arXiv:2504.18599

No real telecom data is used. Works on synthetic data from generate_data.py.
No htm.core dependency required for this module.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. CUSUM Drift Detector
# ─────────────────────────────────────────────

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change point detector.
    Detects gradual drift in streaming time series.
    
    Intuition: accumulates deviations from expected mean.
    When the cumulative sum exceeds a threshold, drift is detected.
    """
    
    def __init__(self, threshold=5.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.mean = None
        self.std = None
        self.warmup_buffer = []
        self.warmup_size = 50
    
    def update(self, value):
        """
        Update CUSUM with new value.
        Returns anomaly score between 0 and 1.
        """
        # Warmup phase: estimate mean and std
        if len(self.warmup_buffer) < self.warmup_size:
            self.warmup_buffer.append(value)
            if len(self.warmup_buffer) == self.warmup_size:
                self.mean = np.mean(self.warmup_buffer)
                self.std = max(np.std(self.warmup_buffer), 1e-6)
            return 0.0
        
        # Normalise
        z = (value - self.mean) / self.std
        
        # Update CUSUM
        self.cusum_pos = max(0, self.cusum_pos + z - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - z - self.drift)
        
        # Score: how far above threshold
        score = max(self.cusum_pos, self.cusum_neg) / self.threshold
        return min(score, 1.0)
    
    def reset(self):
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0


# ─────────────────────────────────────────────
# 2. Z-Score Anomaly Detector
# ─────────────────────────────────────────────

class ZScoreDetector:
    """
    Rolling Z-score anomaly detector.
    Detects sudden spikes and drops.
    
    Intuition: how many standard deviations away from the rolling mean?
    """
    
    def __init__(self, window=100, threshold=3.0):
        self.window = window
        self.threshold = threshold
        self.buffer = []
    
    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        if len(self.buffer) < 10:
            return 0.0
        
        mean = np.mean(self.buffer)
        std = max(np.std(self.buffer), 1e-6)
        z = abs(value - mean) / std
        
        return min(z / self.threshold, 1.0)


# ─────────────────────────────────────────────
# 3. IQR Anomaly Detector
# ─────────────────────────────────────────────

class IQRDetector:
    """
    Interquartile Range (IQR) based outlier detector.
    Robust to non-Gaussian distributions.
    """
    
    def __init__(self, window=200, factor=1.5):
        self.window = window
        self.factor = factor
        self.buffer = []
    
    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        if len(self.buffer) < 20:
            return 0.0
        
        q1 = np.percentile(self.buffer, 25)
        q3 = np.percentile(self.buffer, 75)
        iqr = q3 - q1
        
        lower = q1 - self.factor * iqr
        upper = q3 + self.factor * iqr
        
        if value < lower or value > upper:
            # How far outside the fence?
            distance = max(lower - value, value - upper)
            score = min(distance / (iqr + 1e-6), 1.0)
            return score
        return 0.0


# ─────────────────────────────────────────────
# 4. HTM-Inspired Anomaly Likelihood
# ─────────────────────────────────────────────

class HTMAnomalyLikelihood:
    """
    HTM-inspired anomaly likelihood estimator.
    
    Mimics the AnomalyLikelihood class from htm.core without requiring it.
    Uses a rolling Gaussian model of anomaly scores to estimate
    whether a new score is unusually high.
    
    Based on the approach in Numenta's NAB benchmark.
    """
    
    def __init__(self, window=100, probationary_period=50):
        self.window = window
        self.probationary_period = probationary_period
        self.anomaly_scores = []
        self.count = 0
    
    def update(self, raw_anomaly_score):
        """
        Convert raw anomaly score to anomaly likelihood.
        
        Returns a value between 0 and 1.
        Values > 0.9 indicate likely anomaly.
        """
        self.count += 1
        self.anomaly_scores.append(raw_anomaly_score)
        
        if len(self.anomaly_scores) > self.window:
            self.anomaly_scores.pop(0)
        
        # During probationary period, return raw score
        if self.count < self.probationary_period:
            return raw_anomaly_score
        
        scores = np.array(self.anomaly_scores)
        mean = np.mean(scores)
        std = max(np.std(scores), 1e-6)
        
        # Q-function: probability that score is from normal distribution
        # High likelihood = score is unusually high
        z = (raw_anomaly_score - mean) / std
        likelihood = 1 - stats.norm.cdf(-z)
        
        return float(np.clip(likelihood, 0.0, 1.0))


# ─────────────────────────────────────────────
# 5. Hybrid Detector (main class)
# ─────────────────────────────────────────────

class HybridDriftDetector:
    """
    Hybrid drift and anomaly detector combining statistical methods.
    
    Implements the framework from:
    "A Hybrid Framework for Real-Time Data Drift and Anomaly Identification
    Using Hierarchical Temporal Memory and Statistical Tests"
    
    Components:
    - CUSUM: detects gradual drift
    - Z-score: detects sudden spikes
    - IQR: robust outlier detection
    - HTM anomaly likelihood: calibrated probability estimate
    
    Final score: weighted ensemble of all detectors.
    """
    
    def __init__(
        self,
        cusum_threshold=5.0,
        zscore_window=100,
        zscore_threshold=3.0,
        iqr_window=200,
        weights=None
    ):
        self.cusum = CUSUMDetector(threshold=cusum_threshold)
        self.zscore = ZScoreDetector(window=zscore_window, threshold=zscore_threshold)
        self.iqr = IQRDetector(window=iqr_window)
        self.likelihood = HTMAnomalyLikelihood()
        
        # Weights for ensemble
        self.weights = weights or {
            "cusum": 0.35,
            "zscore": 0.35,
            "iqr": 0.30
        }
        
        self.history = []
    
    def update(self, value):
        """
        Process one new data point.
        Returns dict with individual and combined scores.
        """
        cusum_score = self.cusum.update(value)
        zscore_score = self.zscore.update(value)
        iqr_score = self.iqr.update(value)
        
        # Weighted ensemble
        raw_score = (
            self.weights["cusum"] * cusum_score +
            self.weights["zscore"] * zscore_score +
            self.weights["iqr"] * iqr_score
        )
        
        # HTM-style likelihood
        likelihood = self.likelihood.update(raw_score)
        
        result = {
            "value": value,
            "cusum_score": cusum_score,
            "zscore_score": zscore_score,
            "iqr_score": iqr_score,
            "raw_anomaly_score": raw_score,
            "anomaly_likelihood": likelihood,
            "is_anomaly": likelihood > 0.9
        }
        
        self.history.append(result)
        return result
    
    def process_series(self, values):
        """
        Process a full time series.
        Returns DataFrame with all scores.
        """
        self.history = []
        for v in values:
            self.update(v)
        return pd.DataFrame(self.history)


# ─────────────────────────────────────────────
# 6. Isolation Forest Baseline
# ─────────────────────────────────────────────

def isolation_forest_baseline(values, contamination=0.05, window=200):
    """
    Sliding window Isolation Forest for comparison.
    Batch method - not streaming but useful as baseline.
    """
    values = np.array(values).reshape(-1, 1)
    scaler = StandardScaler()
    
    anomaly_scores = np.zeros(len(values))
    
    for i in range(window, len(values)):
        window_data = values[i-window:i]
        scaled = scaler.fit_transform(window_data)
        
        clf = IsolationForest(
            n_estimators=50,
            contamination=contamination,
            random_state=42
        )
        clf.fit(scaled)
        
        current = scaler.transform(values[i:i+1])
        score = clf.score_samples(current)[0]
        # Convert to 0-1 range (higher = more anomalous)
        anomaly_scores[i] = 1 - (score - clf.offset_) / (clf.offset_ * -1 + 1e-6)
    
    return np.clip(anomaly_scores, 0, 1)


# ─────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────

def evaluate(ground_truth, predictions, threshold=0.5):
    """
    Evaluate anomaly detection performance.
    """
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    binary_preds = (predictions > threshold).astype(int)
    
    tp = np.sum((binary_preds == 1) & (ground_truth == 1))
    fp = np.sum((binary_preds == 1) & (ground_truth == 0))
    fn = np.sum((binary_preds == 0) & (ground_truth == 1))
    tn = np.sum((binary_preds == 0) & (ground_truth == 0))
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (tp + tn) / len(ground_truth)
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn)
    }


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    
    # Generate or load data
    try:
        df = pd.read_csv("data/synthetic_telecom_single.csv")
        print("Loaded existing synthetic data.")
    except FileNotFoundError:
        from generate_data import generate_telecom_timeseries
        df = generate_telecom_timeseries(n_points=2000)
        df.to_csv("data/synthetic_telecom_single.csv", index=False)
        print("Generated synthetic data.")
    
    values = df["SignalSuccessRate"].values
    ground_truth = df["is_anomaly"].values
    
    print(f"\nDataset: {len(values)} points, {ground_truth.sum()} anomaly points")
    
    # Run hybrid detector
    print("\nRunning Hybrid Drift Detector...")
    detector = HybridDriftDetector()
    results_df = detector.process_series(values)
    
    # Evaluate
    metrics = evaluate(
        ground_truth,
        results_df["anomaly_likelihood"].values
    )
    print("\nHybrid Detector Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Run Isolation Forest baseline
    print("\nRunning Isolation Forest baseline...")
    if_scores = isolation_forest_baseline(values)
    if_metrics = evaluate(ground_truth, if_scores)
    print("\nIsolation Forest Results:")
    for k, v in if_metrics.items():
        print(f"  {k}: {v}")
    
    # Save results
    results_df["ground_truth"] = ground_truth
    results_df["if_score"] = if_scores
    results_df.to_csv("data/detection_results.csv", index=False)
    print("\nResults saved to data/detection_results.csv")
