"""
sprt_detector.py

Implementation of the SPRT (Sequential Probability Ratio Test) layer
from the paper:

"A Hybrid Framework for Real-Time Data Drift and Anomaly Identification
Using Hierarchical Temporal Memory and Statistical Tests"
Bandyopadhyay S., Bose J., Roychowdhury S.
IJMEMS 2025. arXiv:2504.18599

The paper proposes:
1. HTM layer outputs anomaly likelihood score at each time step
2. SPRT layer converts HTM output into a binary drift detection decision

This file implements the SPRT layer standalone (no htm.core needed).
It can consume anomaly scores from:
- The HTM pipeline (Part 1 of the notebook, requires htm.core)
- The HybridDriftDetector (Part 2, works without htm.core)

Key insight from the paper (Section 3.2):
HTM output is re-scaled and binarised into a Bernoulli sequence {ct}.
SPRT then tests whether the Bernoulli parameter p has shifted,
indicating a data drift.

This avoids the high false positive rate of KS test, Wasserstein distance,
and PSI — which the paper demonstrates experimentally (Figure 8, 9).
"""

import numpy as np
import pandas as pd
import math


class SPRTDriftDetector:
    """
    Sequential Probability Ratio Test (SPRT) for drift detection.
    
    Implements Theorem 1 from the paper:
    
    Drift detected if:
        Cmt > Upper_limit = [log((1-b)/a) + t * log((1-p_null)/(1-p_alt))]
                            / [log(p_alt/p_null) - log((1-p_alt)/(1-p_null))]
    
    No drift if:
        Cmt < Lower_limit = [log(b/(1-a)) + t * log((1-p_null)/(1-p_alt))]
                            / [log(p_alt/p_null) - log((1-p_alt)/(1-p_null))]
    
    Parameters (from Table 1 in paper):
        p_null: probability threshold for "no drift" hypothesis (default 0.45)
        p_alt: probability threshold for "drift" hypothesis (default 0.50)
        alpha: type 1 error rate (false positive) (default 0.05)
        beta: type 2 error rate (false negative) (default 0.005)
        bin_threshold: threshold to binarise HTM output (default 0.65)
        window_size: rolling window for anomaly score rescaling (default 25)
        anomaly_k: multiplier for rolling std in rescaling (default 1.0)
    
    Usage:
        detector = SPRTDriftDetector()
        for anomaly_score in scores:
            result = detector.update(anomaly_score)
            if result['drift_detected']:
                print(f"Drift at t={result['t']}")
    """
    
    def __init__(
        self,
        p_null=0.45,
        p_alt=0.50,
        alpha=0.05,
        beta=0.005,
        bin_threshold=0.65,
        window_size=25,
        anomaly_k=1.0
    ):
        self.p_null = p_null
        self.p_alt = p_alt
        self.alpha = alpha
        self.beta = beta
        self.bin_threshold = bin_threshold
        self.window_size = window_size
        self.anomaly_k = anomaly_k
        
        # Precompute SPRT constants
        self._log_ratio = (
            math.log(p_alt / p_null) - 
            math.log((1 - p_alt) / (1 - p_null))
        )
        self._log_slope = math.log((1 - p_null) / (1 - p_alt))
        
        # State
        self.t = 0
        self.cmt = 0.0  # Cumulative sum of ct
        self.htm_buffer = []  # Rolling window for rescaling
        self.history = []
        self.drift_count = 0
        
    def _rescale_htm_output(self, htm_value, obs_value):
        """
        Rescale HTM output as per Section 3.4, equations (3) and (4).
        
        anoml_score = |htm_value - obs_value| / (k * rolling_std)
        htmt = min(anoml_score, 1.0)
        
        When htm.core is not available, htm_value IS the anomaly score
        from the HybridDriftDetector, and obs_value is the raw input.
        """
        self.htm_buffer.append(obs_value)
        if len(self.htm_buffer) > self.window_size:
            self.htm_buffer.pop(0)
        
        if len(self.htm_buffer) < 5:
            return htm_value
        
        roll_std = max(np.std(self.htm_buffer), 1e-6)
        
        # Equation (3): normalise by rolling std
        anoml_score = abs(htm_value - np.mean(self.htm_buffer)) / (self.anomaly_k * roll_std)
        
        # Equation (4): clip to [0, 1]
        return min(anoml_score, 1.0)
    
    def _compute_ct(self, htmt):
        """
        Binarise HTM output following Definition 2.
        ct = 1 if htmt > bin_threshold, else 0
        """
        return 1 if htmt > self.bin_threshold else 0
    
    def _compute_limits(self):
        """
        Compute SPRT upper and lower limits from Theorem 1.
        """
        upper = (
            math.log((1 - self.beta) / self.alpha) + 
            self.t * self._log_slope
        ) / self._log_ratio
        
        lower = (
            math.log(self.beta / (1 - self.alpha)) + 
            self.t * self._log_slope
        ) / self._log_ratio
        
        return upper, lower
    
    def update(self, anomaly_score, raw_value=None):
        """
        Process one new anomaly score from HTM or HybridDriftDetector.
        
        Parameters:
            anomaly_score: float in [0,1] from HTM or hybrid detector
            raw_value: optional raw input value for rescaling
        
        Returns dict with SPRT decision and diagnostics.
        """
        self.t += 1
        
        # Rescale if raw value provided
        if raw_value is not None:
            htmt = self._rescale_htm_output(anomaly_score, raw_value)
        else:
            htmt = anomaly_score
        
        # Binarise
        ct = self._compute_ct(htmt)
        
        # Update cumulative sum
        self.cmt += ct
        
        # Compute SPRT limits
        upper_limit, lower_limit = self._compute_limits()
        
        # Decision
        drift_detected = self.cmt > upper_limit
        no_drift = self.cmt < lower_limit
        
        if drift_detected:
            self.drift_count += 1
            # Reset SPRT after drift detection (per paper Section 3.4)
            self.cmt = 0.0
            self.t = 0
        
        result = {
            'step': len(self.history) + 1,
            'anomaly_score': anomaly_score,
            'htmt': htmt,
            'ct': ct,
            'cmt': self.cmt,
            'upper_limit': upper_limit,
            'lower_limit': lower_limit,
            'drift_detected': drift_detected,
            'no_drift': no_drift,
            'inconclusive': not drift_detected and not no_drift
        }
        
        self.history.append(result)
        return result
    
    def process_series(self, anomaly_scores, raw_values=None):
        """
        Process full series of anomaly scores.
        Returns DataFrame with all SPRT decisions.
        """
        self.history = []
        self.t = 0
        self.cmt = 0.0
        self.drift_count = 0
        
        for i, score in enumerate(anomaly_scores):
            raw = raw_values[i] if raw_values is not None else None
            self.update(score, raw)
        
        df = pd.DataFrame(self.history)
        print(f"Total drift events detected: {self.drift_count}")
        return df
    
    def get_drift_windows(self):
        """
        Return list of time points where drift was detected.
        """
        if not self.history:
            return []
        df = pd.DataFrame(self.history)
        return df[df['drift_detected']]['step'].tolist()


class MultivariateSPRT:
    """
    Multivariate extension of SPRT drift detector.
    
    From Section 4 of the paper:
    Run one SPRT per dimension. Combine via majority vote or any rule.
    
    Conservative rule (paper): drift if ANY dimension detects drift.
    """
    
    def __init__(self, n_dims, sprt_params=None, rule='any'):
        """
        Parameters:
            n_dims: number of data dimensions
            sprt_params: dict of SPRT parameters (same for all dims)
            rule: 'any' (drift if any dim drifts) or 'majority'
        """
        self.n_dims = n_dims
        self.rule = rule
        params = sprt_params or {}
        self.detectors = [SPRTDriftDetector(**params) for _ in range(n_dims)]
    
    def update(self, anomaly_scores):
        """
        Update with one vector of anomaly scores (one per dimension).
        Returns combined drift decision.
        """
        results = [
            self.detectors[i].update(anomaly_scores[i])
            for i in range(self.n_dims)
        ]
        
        drift_flags = [r['drift_detected'] for r in results]
        
        if self.rule == 'any':
            combined_drift = any(drift_flags)
        elif self.rule == 'majority':
            combined_drift = sum(drift_flags) > self.n_dims // 2
        else:
            combined_drift = any(drift_flags)
        
        return {
            'drift_detected': combined_drift,
            'drift_by_dim': drift_flags,
            'per_dim_results': results
        }


def compare_drift_methods(values, ground_truth=None, window=25):
    """
    Compare HTM-SPRT against KS test, Wasserstein distance, and PSI.
    Replicates the comparison from Section 3.7 and Figure 8 of the paper.
    
    Returns DataFrame with detection results from all methods.
    """
    from scipy import stats
    from statistical_drift import HybridDriftDetector
    
    n = len(values)
    results = {
        'value': values,
        'ks_drift': np.zeros(n),
        'wasserstein_drift': np.zeros(n),
        'hybrid_sprt_drift': np.zeros(n)
    }
    
    # Wasserstein threshold (simulated 95th percentile)
    w_samples = [
        stats.wasserstein_distance(
            np.random.normal(0, 1, 50),
            np.random.normal(0, 1, 50)
        )
        for _ in range(500)
    ]
    w_threshold = np.percentile(w_samples, 95)
    
    # Rolling window comparison
    for i in range(window * 2, n):
        ref = values[i - window * 2: i - window]
        target = values[i - window: i]
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(ref, target)
        results['ks_drift'][i] = 1 if ks_p < 0.05 else 0
        
        # Wasserstein
        w_dist = stats.wasserstein_distance(ref, target)
        results['wasserstein_drift'][i] = 1 if w_dist > w_threshold else 0
    
    # Hybrid + SPRT
    hybrid = HybridDriftDetector()
    hybrid_results = hybrid.process_series(values)
    
    sprt = SPRTDriftDetector(window_size=window)
    sprt_results = sprt.process_series(
        hybrid_results['anomaly_likelihood'].values,
        raw_values=values
    )
    
    results['hybrid_sprt_drift'] = [
        r['drift_detected'] for r in sprt.history
    ]
    
    df = pd.DataFrame(results)
    
    print(f"\nDrift detection comparison (window={window}):")
    print(f"  KS test detections:          {int(df['ks_drift'].sum())}")
    print(f"  Wasserstein detections:      {int(df['wasserstein_drift'].sum())}")
    print(f"  Hybrid + SPRT detections:    {int(df['hybrid_sprt_drift'].sum())}")
    
    if ground_truth is not None:
        gt = np.array(ground_truth)
        for method in ['ks_drift', 'wasserstein_drift', 'hybrid_sprt_drift']:
            pred = df[method].values
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            print(f"\n  {method}:")
            print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            print(f"    False positives: {fp} (paper shows KS/Wasserstein have very high FP)")
    
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    
    try:
        df = pd.read_csv("data/synthetic_telecom_single.csv")
    except FileNotFoundError:
        from generate_data import generate_telecom_timeseries
        df = generate_telecom_timeseries(n_points=500)
        df.to_csv("data/synthetic_telecom_single.csv", index=False)
    
    values = df["SignalSuccessRate"].values[:500]
    ground_truth = df["is_anomaly"].values[:500]
    
    print("Running SPRT drift detector...")
    print("Parameters from Table 1 (paper): window=25, bin_threshold=0.65, p_null=0.45, p_alt=0.50")
    
    from statistical_drift import HybridDriftDetector
    hybrid = HybridDriftDetector()
    hybrid_df = hybrid.process_series(values)
    
    sprt = SPRTDriftDetector(
        p_null=0.45,
        p_alt=0.50,
        alpha=0.05,
        beta=0.005,
        bin_threshold=0.65,
        window_size=25,
        anomaly_k=1.0
    )
    
    sprt_df = sprt.process_series(
        hybrid_df["anomaly_likelihood"].values,
        raw_values=values
    )
    
    drift_points = sprt.get_drift_windows()
    print(f"\nDrift detected at time points: {drift_points}")
    
    print("\nRunning method comparison (replicates paper Figure 8)...")
    comparison_df = compare_drift_methods(values, ground_truth, window=25)
    comparison_df.to_csv("data/drift_comparison.csv", index=False)
    print("\nComparison saved to data/drift_comparison.csv")
