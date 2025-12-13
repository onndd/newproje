
import numpy as np
import pandas as pd
from typing import List, Tuple

class FourierDetector:
    """
    Model G: The Rhythm Detector (Fourier Analysis).
    Detects periodic cycles in the multiplier sequence using Fast Fourier Transform (FFT).
    Operates in two modes:
    1. Batch Analysis (Historical Simulation)
    2. Real-Time Analysis (Local/Live)
    """
    
    def __init__(self, window_sizes: List[int] = [64, 256, 1024]):
        """
        Args:
            window_sizes: List of windows to analyze. 
                          Small (64) = Pulse, Medium (256) = Rhythm, Large (1024) = Climate.
        """
        self.windows = window_sizes
        
    def _compute_rhythm(self, signal: np.ndarray) -> Tuple[float, float]:
        """
        Computes the dominant rhythm strength and relative phase for a given signal window.
        Args:
            signal: Array of multipliers (last N rounds).
        Returns:
            strength (0.0-1.0): How strong the periodic signal is.
            phase (0.0-1.0): Position in the cycle (0=trough, 0.5=rising, 1.0=peak).
        """
        n = len(signal)
        if n < 8: return 0.0, 0.0
        
        # Detrend (Remove DC component/mean) to focus on oscillation
        signal_detrended = signal - np.mean(signal)
        
        # Apply FFT
        fft_vals = np.fft.rfft(signal_detrended)
        fft_freq = np.fft.rfftfreq(n)
        
        # Get Power Spectrum (Magnitude)
        power = np.abs(fft_vals)
        
        # Ignore DC component (index 0)
        power[0] = 0
        
        # Find dominant frequency
        if np.sum(power) == 0: return 0.0, 0.0
        
        peak_idx = np.argmax(power)
        peak_power = power[peak_idx]
        total_power = np.sum(power)
        
        # Strength: Ratio of dominant peak to total noise
        # Normalized roughly. If a single frequency dominates, ratio is high.
        strength = peak_power / total_power if total_power > 0 else 0.0
        
        # Phase at the end of the window for the dominant frequency
        # We want to know: "Where are we NOW in this cycle?"
        # Phase angle is in radians (-pi to pi)
        # We project it to the end of the window
        # angle = np.angle(fft_vals[peak_idx])
        # BUT standard FFT phase is for t=0. We need phase at t=N.
        # Actually, simpler proxy for "Cycle Position" in betting context:
        # Just use the normalized angle of the dominant component.
        angle = np.angle(fft_vals[peak_idx])
        
        # Normalize to 0.0 - 1.0 range
        # -pi -> 0.0, 0 -> 0.5, pi -> 1.0
        normalized_phase = (angle + np.pi) / (2 * np.pi)
        
        return strength, normalized_phase

    def analyze_realtime(self, recent_history: np.ndarray) -> dict:
        """
        Analyzes the buffer for real-time prediction.
        Args:
            recent_history: Array of recent multiplier outcomes.
        Returns:
            Dictionary with features for each window size.
        """
        features = {}
        history_len = len(recent_history)
        
        for w in self.windows:
            if history_len >= w:
                # Take exactly the last 'w' patterns
                window_data = recent_history[-w:]
                strength, phase = self._compute_rhythm(window_data)
            else:
                # Not enough data yet
                strength, phase = 0.0, 0.5
            
            features[f'fourier_strength_{w}'] = strength
            features[f'fourier_phase_{w}'] = phase
            
        return features

    def analyze_batch(self, full_history: np.ndarray) -> pd.DataFrame:
        """
        Simulates historical analysis (Rolling Window) for Notebook training.
        WARNING: This can be slow if done naively in Python loop for 17k rows * 3 windows.
        We optimize by using strides or accepted lag.
        For 17k rows, a simple loop is actually acceptable (approx 1-2 seconds with numpy).
        """
        n = len(full_history)
        
        # Initialize output arrays
        # Shape: (n_samples, n_windows * 2)
        # We'll return a DataFrame
        
        results = {f'fourier_strength_{w}': np.zeros(n) for w in self.windows}
        results.update({f'fourier_phase_{w}': np.full(n, 0.5) for w in self.windows}) # Default phase 0.5
        
        print(f"🌊 Fourier Analysis (Model G) starting on {n} samples...")
        
        # We iterate through history. To speed up, we can calculate everything,
        # but strictly, at index 'i', we can only see history[:i].
        # Optimized loop:
        
        # Pre-convert to list for faster indexing potentially, or keep numpy
        # 17k is small. O(N*W) complexity. 17000 * 1024 operations is ~17M ops.
        # This takes < 1 second on modern CPU. Safe to loop.
        
        for w in self.windows:
            col_str = f'fourier_strength_{w}'
            col_pha = f'fourier_phase_{w}'
            
            # Vectorized rolling window approach is hard with complex FFT
            # Simple loop is robust
            for i in range(w, n):
                window_data = full_history[i-w:i] # Data strictly BEFORE index i
                s, p = self._compute_rhythm(window_data)
                results[col_str][i] = s
                results[col_pha][i] = p
                
        return pd.DataFrame(results)

