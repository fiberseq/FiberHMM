"""
FiberHMM HMM module

Provides:
1. Native 2-state HMM implementation (no dependencies beyond numpy/scipy)
2. Viterbi decoding, forward-backward, and Baum-Welch training
3. Optional Numba JIT compilation for ~10x speedup

Model I/O (load/save) has been moved to fiberhmm.core.model_io.
For backward compatibility, load_model, save_model, and
load_model_with_metadata are re-exported from this module.
"""

import warnings
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import json

# Use tqdm for progress bars if available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Use scipy's logsumexp if available (faster C implementation)
try:
    from scipy.special import logsumexp as scipy_logsumexp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Use numba for JIT compilation if available
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# Numba JIT-compiled HMM algorithms (much faster than pure Python)
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=False)
    def _viterbi_numba(obs, log_startprob, log_transmat, log_emissionprob):
        """
        Numba-compiled Viterbi for 2-state HMM.
        
        Args:
            obs: Observation sequence (int array)
            log_startprob: (2,) log start probabilities
            log_transmat: (2, 2) log transition matrix
            log_emissionprob: (2, n_symbols) log emission probabilities
            
        Returns:
            path: Most likely state sequence
            log_prob: Log probability of path
        """
        T = len(obs)
        
        # Extract transition probabilities
        log_trans_00 = log_transmat[0, 0]
        log_trans_01 = log_transmat[0, 1]
        log_trans_10 = log_transmat[1, 0]
        log_trans_11 = log_transmat[1, 1]
        
        # Viterbi arrays
        viterbi_0 = np.empty(T)
        viterbi_1 = np.empty(T)
        backpointer_0 = np.zeros(T, dtype=np.int8)
        backpointer_1 = np.zeros(T, dtype=np.int8)
        
        # Initialize
        viterbi_0[0] = log_startprob[0] + log_emissionprob[0, obs[0]]
        viterbi_1[0] = log_startprob[1] + log_emissionprob[1, obs[0]]
        
        # Forward pass
        for t in range(1, T):
            o = obs[t]
            v0_prev = viterbi_0[t-1]
            v1_prev = viterbi_1[t-1]
            
            emit_0 = log_emissionprob[0, o]
            emit_1 = log_emissionprob[1, o]
            
            # State 0
            from_0_to_0 = v0_prev + log_trans_00
            from_1_to_0 = v1_prev + log_trans_10
            if from_0_to_0 >= from_1_to_0:
                viterbi_0[t] = from_0_to_0 + emit_0
                backpointer_0[t] = 0
            else:
                viterbi_0[t] = from_1_to_0 + emit_0
                backpointer_0[t] = 1
            
            # State 1
            from_0_to_1 = v0_prev + log_trans_01
            from_1_to_1 = v1_prev + log_trans_11
            if from_0_to_1 >= from_1_to_1:
                viterbi_1[t] = from_0_to_1 + emit_1
                backpointer_1[t] = 0
            else:
                viterbi_1[t] = from_1_to_1 + emit_1
                backpointer_1[t] = 1
        
        # Backtrack
        path = np.zeros(T, dtype=np.int8)
        if viterbi_0[T-1] >= viterbi_1[T-1]:
            path[T-1] = 0
            log_prob = viterbi_0[T-1]
        else:
            path[T-1] = 1
            log_prob = viterbi_1[T-1]
        
        for t in range(T - 2, -1, -1):
            if path[t + 1] == 0:
                path[t] = backpointer_0[t + 1]
            else:
                path[t] = backpointer_1[t + 1]
        
        return path, log_prob
    
    @jit(nopython=True, cache=False)
    def _forward_numba(obs, log_startprob, log_transmat, log_emissionprob):
        """Numba-compiled forward algorithm for 2-state HMM."""
        T = len(obs)
        alpha = np.empty((T, 2))
        
        log_trans_00 = log_transmat[0, 0]
        log_trans_01 = log_transmat[0, 1]
        log_trans_10 = log_transmat[1, 0]
        log_trans_11 = log_transmat[1, 1]
        
        # Initialize
        alpha[0, 0] = log_startprob[0] + log_emissionprob[0, obs[0]]
        alpha[0, 1] = log_startprob[1] + log_emissionprob[1, obs[0]]
        
        # Forward pass
        for t in range(1, T):
            o = obs[t]
            a0_prev = alpha[t-1, 0]
            a1_prev = alpha[t-1, 1]
            
            # State 0: logsumexp(a0 + trans_00, a1 + trans_10)
            x0 = a0_prev + log_trans_00
            y0 = a1_prev + log_trans_10
            if x0 >= y0:
                alpha[t, 0] = x0 + np.log1p(np.exp(y0 - x0)) + log_emissionprob[0, o]
            else:
                alpha[t, 0] = y0 + np.log1p(np.exp(x0 - y0)) + log_emissionprob[0, o]
            
            # State 1: logsumexp(a0 + trans_01, a1 + trans_11)
            x1 = a0_prev + log_trans_01
            y1 = a1_prev + log_trans_11
            if x1 >= y1:
                alpha[t, 1] = x1 + np.log1p(np.exp(y1 - x1)) + log_emissionprob[1, o]
            else:
                alpha[t, 1] = y1 + np.log1p(np.exp(x1 - y1)) + log_emissionprob[1, o]
        
        # Final log prob
        a0_final = alpha[T-1, 0]
        a1_final = alpha[T-1, 1]
        if a0_final >= a1_final:
            log_prob = a0_final + np.log1p(np.exp(a1_final - a0_final))
        else:
            log_prob = a1_final + np.log1p(np.exp(a0_final - a1_final))
        
        return alpha, log_prob
    
    @jit(nopython=True, cache=False)
    def _backward_numba(obs, log_transmat, log_emissionprob):
        """Numba-compiled backward algorithm for 2-state HMM."""
        T = len(obs)
        beta = np.empty((T, 2))
        
        log_trans_00 = log_transmat[0, 0]
        log_trans_01 = log_transmat[0, 1]
        log_trans_10 = log_transmat[1, 0]
        log_trans_11 = log_transmat[1, 1]
        
        # Initialize
        beta[T-1, 0] = 0.0
        beta[T-1, 1] = 0.0
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            o = obs[t + 1]
            b0_next = beta[t+1, 0]
            b1_next = beta[t+1, 1]
            emit_0 = log_emissionprob[0, o]
            emit_1 = log_emissionprob[1, o]
            
            # beta[t, 0] = logsumexp(trans_00 + emit_0 + b0, trans_01 + emit_1 + b1)
            x0 = log_trans_00 + emit_0 + b0_next
            y0 = log_trans_01 + emit_1 + b1_next
            if x0 >= y0:
                beta[t, 0] = x0 + np.log1p(np.exp(y0 - x0))
            else:
                beta[t, 0] = y0 + np.log1p(np.exp(x0 - y0))
            
            # beta[t, 1] = logsumexp(trans_10 + emit_0 + b0, trans_11 + emit_1 + b1)
            x1 = log_trans_10 + emit_0 + b0_next
            y1 = log_trans_11 + emit_1 + b1_next
            if x1 >= y1:
                beta[t, 1] = x1 + np.log1p(np.exp(y1 - x1))
            else:
                beta[t, 1] = y1 + np.log1p(np.exp(x1 - y1))
        
        return beta
    
    @jit(nopython=True, cache=False)
    def _baum_welch_estep_numba(obs, log_startprob, log_transmat, log_emissionprob):
        """
        Numba-compiled full E-step: forward, backward, and accumulate counts.
        
        Returns:
            start_counts: (2,) start state counts
            trans_counts: (2, 2) transition counts
            log_prob: log probability of sequence
        """
        T = len(obs)
        
        # Extract transition probabilities for speed
        log_trans_00 = log_transmat[0, 0]
        log_trans_01 = log_transmat[0, 1]
        log_trans_10 = log_transmat[1, 0]
        log_trans_11 = log_transmat[1, 1]
        
        # Forward pass
        alpha = np.empty((T, 2))
        alpha[0, 0] = log_startprob[0] + log_emissionprob[0, obs[0]]
        alpha[0, 1] = log_startprob[1] + log_emissionprob[1, obs[0]]
        
        for t in range(1, T):
            o = obs[t]
            a0_prev = alpha[t-1, 0]
            a1_prev = alpha[t-1, 1]
            
            # alpha[t, 0] = logsumexp(trans_00 + a0, trans_10 + a1) + emit
            x0 = log_trans_00 + a0_prev
            y0 = log_trans_10 + a1_prev
            if x0 >= y0:
                alpha[t, 0] = x0 + np.log1p(np.exp(y0 - x0)) + log_emissionprob[0, o]
            else:
                alpha[t, 0] = y0 + np.log1p(np.exp(x0 - y0)) + log_emissionprob[0, o]
            
            x1 = log_trans_01 + a0_prev
            y1 = log_trans_11 + a1_prev
            if x1 >= y1:
                alpha[t, 1] = x1 + np.log1p(np.exp(y1 - x1)) + log_emissionprob[1, o]
            else:
                alpha[t, 1] = y1 + np.log1p(np.exp(x1 - y1)) + log_emissionprob[1, o]
        
        # Final log prob
        a0_final = alpha[T-1, 0]
        a1_final = alpha[T-1, 1]
        if a0_final >= a1_final:
            log_prob = a0_final + np.log1p(np.exp(a1_final - a0_final))
        else:
            log_prob = a1_final + np.log1p(np.exp(a0_final - a1_final))
        
        # Backward pass
        beta = np.empty((T, 2))
        beta[T-1, 0] = 0.0
        beta[T-1, 1] = 0.0
        
        for t in range(T - 2, -1, -1):
            o = obs[t + 1]
            b0_next = beta[t+1, 0]
            b1_next = beta[t+1, 1]
            emit_0 = log_emissionprob[0, o]
            emit_1 = log_emissionprob[1, o]
            
            x0 = log_trans_00 + emit_0 + b0_next
            y0 = log_trans_01 + emit_1 + b1_next
            if x0 >= y0:
                beta[t, 0] = x0 + np.log1p(np.exp(y0 - x0))
            else:
                beta[t, 0] = y0 + np.log1p(np.exp(x0 - y0))
            
            x1 = log_trans_10 + emit_0 + b0_next
            y1 = log_trans_11 + emit_1 + b1_next
            if x1 >= y1:
                beta[t, 1] = x1 + np.log1p(np.exp(y1 - x1))
            else:
                beta[t, 1] = y1 + np.log1p(np.exp(x1 - y1))
        
        # Compute gamma for start counts
        gamma_0 = alpha[0, 0] + beta[0, 0] - log_prob
        gamma_1 = alpha[0, 1] + beta[0, 1] - log_prob
        start_counts = np.array([np.exp(gamma_0), np.exp(gamma_1)])
        
        # Compute transition counts
        trans_counts = np.zeros((2, 2))
        for t in range(T - 1):
            o_next = obs[t + 1]
            emit_0 = log_emissionprob[0, o_next]
            emit_1 = log_emissionprob[1, o_next]
            
            xi_00 = np.exp(alpha[t, 0] + log_trans_00 + emit_0 + beta[t+1, 0] - log_prob)
            xi_01 = np.exp(alpha[t, 0] + log_trans_01 + emit_1 + beta[t+1, 1] - log_prob)
            xi_10 = np.exp(alpha[t, 1] + log_trans_10 + emit_0 + beta[t+1, 0] - log_prob)
            xi_11 = np.exp(alpha[t, 1] + log_trans_11 + emit_1 + beta[t+1, 1] - log_prob)
            
            trans_counts[0, 0] += xi_00
            trans_counts[0, 1] += xi_01
            trans_counts[1, 0] += xi_10
            trans_counts[1, 1] += xi_11
        
        return start_counts, trans_counts, log_prob


class FiberHMM:
    """
    Simple 2-state HMM for FiberHMM footprint calling.
    
    States:
        0: Footprint/Inaccessible (low methylation probability)
        1: Accessible (high methylation probability)
    
    This implementation uses log probabilities throughout for numerical stability.
    """
    
    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.startprob_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.emissionprob_: Optional[np.ndarray] = None
        
        # Log versions (computed when needed)
        self._log_startprob: Optional[np.ndarray] = None
        self._log_transmat: Optional[np.ndarray] = None
        self._log_emissionprob: Optional[np.ndarray] = None
        
        # Training metadata
        self.n_iter: int = 1000  # Max EM iterations (usually converges much faster)
        self.tol: float = 1e-4
        self.monitor_: Optional[TrainingMonitor] = None
    
    def _compute_log_probs(self):
        """Convert probabilities to log space."""
        with np.errstate(divide='ignore'):  # Handle log(0) gracefully
            if self.startprob_ is not None:
                self._log_startprob = np.log(self.startprob_)
            if self.transmat_ is not None:
                self._log_transmat = np.log(self.transmat_)
            if self.emissionprob_ is not None:
                self._log_emissionprob = np.log(self.emissionprob_)
    
    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm in log space, optimized for 2-state HMM.
        
        Returns:
            alpha: Forward probabilities (T x 2)
            log_prob: Log probability of observation sequence
        """
        # Use Numba-compiled version if available
        if HAS_NUMBA:
            return _forward_numba(obs, self._log_startprob, self._log_transmat,
                                   self._log_emissionprob)
        
        # Fallback to pure Python
        T = len(obs)
        alpha = np.empty((T, 2))
        
        log_trans_00 = self._log_transmat[0, 0]
        log_trans_01 = self._log_transmat[0, 1]
        log_trans_10 = self._log_transmat[1, 0]
        log_trans_11 = self._log_transmat[1, 1]
        log_emit = self._log_emissionprob
        
        alpha[0, 0] = self._log_startprob[0] + log_emit[0, obs[0]]
        alpha[0, 1] = self._log_startprob[1] + log_emit[1, obs[0]]
        
        for t in range(1, T):
            o = obs[t]
            a0_prev = alpha[t-1, 0]
            a1_prev = alpha[t-1, 1]
            
            x0 = a0_prev + log_trans_00
            y0 = a1_prev + log_trans_10
            if x0 >= y0:
                alpha[t, 0] = x0 + np.log1p(np.exp(y0 - x0)) + log_emit[0, o]
            else:
                alpha[t, 0] = y0 + np.log1p(np.exp(x0 - y0)) + log_emit[0, o]
            
            x1 = a0_prev + log_trans_01
            y1 = a1_prev + log_trans_11
            if x1 >= y1:
                alpha[t, 1] = x1 + np.log1p(np.exp(y1 - x1)) + log_emit[1, o]
            else:
                alpha[t, 1] = y1 + np.log1p(np.exp(x1 - y1)) + log_emit[1, o]
        
        a0_final = alpha[-1, 0]
        a1_final = alpha[-1, 1]
        if a0_final >= a1_final:
            log_prob = a0_final + np.log1p(np.exp(a1_final - a0_final))
        else:
            log_prob = a1_final + np.log1p(np.exp(a0_final - a1_final))
        
        return alpha, log_prob
    
    def _backward(self, obs: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log space, optimized for 2-state HMM.
        
        Returns:
            beta: Backward probabilities (T x 2)
        """
        # Use Numba-compiled version if available
        if HAS_NUMBA:
            return _backward_numba(obs, self._log_transmat, self._log_emissionprob)
        
        # Fallback to pure Python
        T = len(obs)
        beta = np.empty((T, 2))
        
        log_trans_00 = self._log_transmat[0, 0]
        log_trans_01 = self._log_transmat[0, 1]
        log_trans_10 = self._log_transmat[1, 0]
        log_trans_11 = self._log_transmat[1, 1]
        log_emit = self._log_emissionprob
        
        beta[-1, 0] = 0.0
        beta[-1, 1] = 0.0
        
        for t in range(T - 2, -1, -1):
            o = obs[t + 1]
            b0_next = beta[t+1, 0]
            b1_next = beta[t+1, 1]
            emit_0 = log_emit[0, o]
            emit_1 = log_emit[1, o]
            
            x0 = log_trans_00 + emit_0 + b0_next
            y0 = log_trans_01 + emit_1 + b1_next
            if x0 >= y0:
                beta[t, 0] = x0 + np.log1p(np.exp(y0 - x0))
            else:
                beta[t, 0] = y0 + np.log1p(np.exp(x0 - y0))
            
            x1 = log_trans_10 + emit_0 + b0_next
            y1 = log_trans_11 + emit_1 + b1_next
            if x1 >= y1:
                beta[t, 1] = x1 + np.log1p(np.exp(y1 - x1))
            else:
                beta[t, 1] = y1 + np.log1p(np.exp(x1 - y1))
        
        return beta
    
    def _viterbi(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm for most likely state sequence.
        
        Returns:
            path: Most likely state sequence
            log_prob: Log probability of the path
        """
        # Use Numba-compiled version if available
        if HAS_NUMBA:
            return _viterbi_numba(obs, self._log_startprob, self._log_transmat, 
                                   self._log_emissionprob)
        
        # Fallback to pure Python
        T = len(obs)
        
        log_trans_00 = self._log_transmat[0, 0]
        log_trans_01 = self._log_transmat[0, 1]
        log_trans_10 = self._log_transmat[1, 0]
        log_trans_11 = self._log_transmat[1, 1]
        log_emit = self._log_emissionprob
        
        viterbi_0 = np.empty(T)
        viterbi_1 = np.empty(T)
        backpointer_0 = np.zeros(T, dtype=np.int8)
        backpointer_1 = np.zeros(T, dtype=np.int8)
        
        viterbi_0[0] = self._log_startprob[0] + log_emit[0, obs[0]]
        viterbi_1[0] = self._log_startprob[1] + log_emit[1, obs[0]]
        
        for t in range(1, T):
            o = obs[t]
            v0_prev = viterbi_0[t-1]
            v1_prev = viterbi_1[t-1]
            
            from_0_to_0 = v0_prev + log_trans_00
            from_1_to_0 = v1_prev + log_trans_10
            if from_0_to_0 >= from_1_to_0:
                viterbi_0[t] = from_0_to_0 + log_emit[0, o]
                backpointer_0[t] = 0
            else:
                viterbi_0[t] = from_1_to_0 + log_emit[0, o]
                backpointer_0[t] = 1
            
            from_0_to_1 = v0_prev + log_trans_01
            from_1_to_1 = v1_prev + log_trans_11
            if from_0_to_1 >= from_1_to_1:
                viterbi_1[t] = from_0_to_1 + log_emit[1, o]
                backpointer_1[t] = 0
            else:
                viterbi_1[t] = from_1_to_1 + log_emit[1, o]
                backpointer_1[t] = 1
        
        path = np.zeros(T, dtype=np.int8)
        if viterbi_0[-1] >= viterbi_1[-1]:
            path[-1] = 0
            log_prob = viterbi_0[-1]
        else:
            path[-1] = 1
            log_prob = viterbi_1[-1]
        
        for t in range(T - 2, -1, -1):
            if path[t + 1] == 0:
                path[t] = backpointer_0[t + 1]
            else:
                path[t] = backpointer_1[t + 1]
        
        return path, log_prob
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        
        Args:
            X: Observation sequence, shape (T, 1) or (T,)
            
        Returns:
            State sequence, shape (T,)
        """
        self._compute_log_probs()
        
        obs = X.flatten().astype(int)
        path, _ = self._viterbi(obs)
        return path
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities P(state | observations) at each position.
        
        Uses the forward-backward algorithm.
        
        Args:
            X: Observation sequence, shape (T, 1) or (T,)
            
        Returns:
            Posterior probabilities, shape (T, n_states)
            Each row sums to 1.0
        """
        self._compute_log_probs()
        
        obs = X.flatten().astype(int)
        
        # Forward-backward
        alpha, log_prob = self._forward(obs)
        beta = self._backward(obs)
        
        # Compute gamma (posteriors) in log space
        log_gamma = alpha + beta
        
        # Normalize each row (subtract logsumexp to get proper probabilities)
        log_gamma -= _logsumexp_axis1(log_gamma)[:, np.newaxis]
        
        # Convert from log space
        gamma = np.exp(log_gamma)
        
        return gamma
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state sequence and return confidence scores.
        
        Combines Viterbi decoding with posterior probabilities to give
        both the most likely path and a confidence score for each position.
        
        Args:
            X: Observation sequence, shape (T, 1) or (T,)
            
        Returns:
            path: Most likely state sequence, shape (T,)
            confidence: P(predicted_state | observations), shape (T,)
        """
        self._compute_log_probs()
        
        obs = X.flatten().astype(int)
        
        # Get Viterbi path
        path, _ = self._viterbi(obs)
        
        # Get posteriors
        alpha, log_prob = self._forward(obs)
        beta = self._backward(obs)
        
        log_gamma = alpha + beta
        log_gamma -= _logsumexp_axis1(log_gamma)[:, np.newaxis]
        gamma = np.exp(log_gamma)
        
        # Confidence is the posterior probability of the predicted state
        T = len(path)
        confidence = gamma[np.arange(T), path]
        
        return path, confidence
    
    def predict_with_posteriors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state sequence and return full posterior matrix.
        
        More efficient than calling predict() and predict_proba() separately
        since we share the forward-backward computation.
        
        Args:
            X: Observation sequence, shape (T, 1) or (T,)
            
        Returns:
            path: Most likely state sequence, shape (T,)
            posteriors: P(state | observations), shape (T, n_states)
        """
        self._compute_log_probs()
        
        obs = X.flatten().astype(int)
        
        # Get Viterbi path
        path, _ = self._viterbi(obs)
        
        # Get posteriors via forward-backward
        alpha, log_prob = self._forward(obs)
        beta = self._backward(obs)
        
        log_gamma = alpha + beta
        log_gamma -= _logsumexp_axis1(log_gamma)[:, np.newaxis]
        posteriors = np.exp(log_gamma)
        
        return path, posteriors
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log probability of observation sequence.
        
        Args:
            X: Observation sequence
            
        Returns:
            Log probability
        """
        self._compute_log_probs()
        obs = X.flatten().astype(int)
        _, log_prob = self._forward(obs)
        return log_prob
    
    def fit(self, X: np.ndarray, lengths: Optional[list] = None,
            verbose: bool = False, desc: str = "EM") -> 'FiberHMM':
        """
        Train HMM using Baum-Welch algorithm.
        
        Only trains start and transition probabilities.
        Emission probabilities are fixed.
        
        Args:
            X: Observation sequence(s), shape (T, 1)
            lengths: Length of each sequence if multiple concatenated
            verbose: Show progress bar for EM iterations
            desc: Description for progress bar
            
        Returns:
            self
        """
        self._compute_log_probs()
        
        obs = X.flatten().astype(int)
        
        if lengths is None:
            lengths = [len(obs)]
        
        self.monitor_ = TrainingMonitor()
        prev_log_prob = -np.inf
        
        # Check if we can use numba-optimized E-step
        use_numba_estep = HAS_NUMBA and len(lengths) == 1
        
        iterator = range(self.n_iter)
        if verbose and HAS_TQDM:
            iterator = tqdm(iterator, desc=desc, leave=False)
        
        for iteration in iterator:
            # E-step: compute expected counts
            log_prob_total = 0.0
            start_counts = np.zeros(self.n_states)
            trans_counts = np.zeros((self.n_states, self.n_states))
            
            if use_numba_estep:
                # Use optimized numba version for single sequence
                sc, tc, lp = _baum_welch_estep_numba(
                    obs, self._log_startprob, self._log_transmat, self._log_emissionprob
                )
                start_counts = sc
                trans_counts = tc
                log_prob_total = lp
            else:
                # Fall back to sequence-by-sequence processing
                idx = 0
                for length in lengths:
                    seq = obs[idx:idx + length]
                    idx += length
                    
                    if HAS_NUMBA:
                        # Use numba E-step for this sequence
                        sc, tc, lp = _baum_welch_estep_numba(
                            seq, self._log_startprob, self._log_transmat, self._log_emissionprob
                        )
                        start_counts += sc
                        trans_counts += tc
                        log_prob_total += lp
                    else:
                        # Pure Python fallback
                        alpha, log_prob = self._forward(seq)
                        beta = self._backward(seq)
                        log_prob_total += log_prob
                        
                        # Posterior probabilities
                        gamma = alpha + beta
                        gamma -= _logsumexp(gamma, axis=1, keepdims=True)
                        gamma = np.exp(gamma)
                        
                        # Start counts
                        start_counts += gamma[0]
                        
                        # Transition counts - vectorized
                        for t in range(len(seq) - 1):
                            log_xi = (alpha[t, :, np.newaxis] +
                                     self._log_transmat +
                                     self._log_emissionprob[:, seq[t+1]] +
                                     beta[t+1, :])
                            log_xi -= log_prob
                            trans_counts += np.exp(log_xi)
            
            # M-step: update parameters
            self.startprob_ = start_counts / start_counts.sum()
            self.startprob_ = np.clip(self.startprob_, 1e-10, 1.0)
            self.startprob_ /= self.startprob_.sum()
            
            trans_sums = trans_counts.sum(axis=1, keepdims=True)
            trans_sums = np.where(trans_sums == 0, 1, trans_sums)
            self.transmat_ = trans_counts / trans_sums
            self.transmat_ = np.clip(self.transmat_, 1e-10, 1.0)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            
            self._compute_log_probs()
            
            # Check convergence
            self.monitor_.history.append(log_prob_total)
            
            improvement = log_prob_total - prev_log_prob
            
            # Update progress bar
            if verbose and HAS_TQDM and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'logprob': f'{log_prob_total:.2e}', 
                                     'delta': f'{improvement:.2e}'})
            
            if iteration > 0 and improvement < self.tol:
                break
            
            prev_log_prob = log_prob_total
        
        return self
    
    def normalize_states(self, verbose: bool = False) -> bool:
        """
        Ensure states are in canonical order:
        - State 0: Footprint (LOW methylation probability)
        - State 1: Accessible (HIGH methylation probability)
        
        The HMM training (Baum-Welch) can converge to either state assignment.
        This method detects and fixes reversed states by computing mean emission
        probabilities for the methylated positions (first half of emission matrix).
        
        Note: The emission matrix has format [methylated_codes, unmethylated_codes]
        where unmethylated = 1 - methylated. We only check the first half since
        the second half would average out to similar values.
        
        Args:
            verbose: If True, print diagnostic information
            
        Returns:
            True if states were swapped, False otherwise
        """
        if self.emissionprob_ is None:
            return False
        
        # Only look at the first half (methylated positions)
        # The emission matrix has format: [methylated, unmethylated] where unmeth = 1 - meth
        # Looking at the full matrix gives ~0.5 for both states, which is useless
        n_obs = self.emissionprob_.shape[1]
        n_methylated = n_obs // 2  # First half is methylated
        
        # Compute mean emission probability for each state (methylated positions only)
        # Lower mean = lower methylation probability = footprint state (State 0)
        # Higher mean = higher methylation probability = accessible state (State 1)
        mean_0 = np.mean(self.emissionprob_[0, :n_methylated])
        mean_1 = np.mean(self.emissionprob_[1, :n_methylated])
        
        if verbose:
            print(f"Methylated emission means: State 0 = {mean_0:.4f}, State 1 = {mean_1:.4f}")
        
        # States are correct if State 0 has LOWER mean (footprint)
        if mean_0 <= mean_1:
            if verbose:
                print("States are in correct order (State 0 = footprint, State 1 = accessible)")
            return False
        
        # States are reversed - swap them
        if verbose:
            print("States are REVERSED - swapping (State 0 had higher mean, should be footprint)")
        
        # Swap emission probabilities
        self.emissionprob_ = self.emissionprob_[[1, 0], :]
        
        # Swap start probabilities
        if self.startprob_ is not None:
            self.startprob_ = self.startprob_[[1, 0]]
        
        # Swap transition matrix rows and columns
        if self.transmat_ is not None:
            # Swap rows, then swap columns
            self.transmat_ = self.transmat_[[1, 0], :][:, [1, 0]]
        
        # Clear cached log probabilities (they need to be recomputed)
        self._log_startprob = None
        self._log_transmat = None
        self._log_emissionprob = None
        
        if verbose:
            new_mean_0 = np.mean(self.emissionprob_[0, :n_methylated])
            new_mean_1 = np.mean(self.emissionprob_[1, :n_methylated])
            print(f"After swap: State 0 = {new_mean_0:.4f}, State 1 = {new_mean_1:.4f}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            'n_states': self.n_states,
            'startprob_': self.startprob_.tolist() if self.startprob_ is not None else None,
            'transmat_': self.transmat_.tolist() if self.transmat_ is not None else None,
            'emissionprob_': self.emissionprob_.tolist() if self.emissionprob_ is not None else None,
            'model_type': 'FiberHMM_native'
        }
    
    def to_json_rust(self, filepath: str):
        """
        Export model in JSON format compatible with Rust fiberhmm.
        
        The Rust version expects:
        - startprob: [f64; 2]
        - transmat: [[f64; 2]; 2]
        - emissionprob: Vec<[f64; 2]> - list of [accessible_prob, footprint_prob] per symbol
        """
        # Transpose emission probs: Python is (n_states, n_symbols), Rust wants (n_symbols, n_states)
        emit_transposed = self.emissionprob_.T.tolist()  # Now (n_symbols, 2)
        
        data = {
            'startprob': self.startprob_.tolist(),
            'transmat': self.transmat_.tolist(),
            'emissionprob': emit_transposed,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported Rust-compatible model to {filepath}")
        print(f"  Symbols: {len(emit_transposed)}")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FiberHMM':
        """Deserialize model from dictionary."""
        model = cls(n_states=d.get('n_states', 2))
        if d.get('startprob_') is not None:
            model.startprob_ = np.array(d['startprob_'])
        if d.get('transmat_') is not None:
            model.transmat_ = np.array(d['transmat_'])
        if d.get('emissionprob_') is not None:
            model.emissionprob_ = np.array(d['emissionprob_'])
        return model


class TrainingMonitor:
    """Tracks training progress."""
    def __init__(self):
        self.history = []


def _logsumexp(a: np.ndarray, axis: Optional[int] = None, 
               keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp. Uses scipy if available."""
    if HAS_SCIPY:
        return scipy_logsumexp(a, axis=axis, keepdims=keepdims)
    
    # Fallback numpy implementation
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0)
    result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims)) + a_max.squeeze()
    
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    
    return result


def _logsumexp_axis1(a: np.ndarray) -> np.ndarray:
    """Logsumexp along axis 1, optimized for 2D arrays."""
    a_max = np.max(a, axis=1, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0)
    return np.log(np.sum(np.exp(a - a_max), axis=1)) + a_max.flatten()


# =============================================================================
# HMM model creation and training
# =============================================================================

def create_hmmlearn_model(emission_probs: np.ndarray,
                          use_legacy: bool = False) -> Union[FiberHMM, Any]:
    """
    Create an HMM model for training.

    Args:
        emission_probs: Emission probability matrix (n_states x n_observations)
        use_legacy: If True, try to use hmmlearn (for reproducibility with old results)

    Returns:
        FiberHMM (native) or hmmlearn model
    """
    if use_legacy:
        model = _try_create_hmmlearn(emission_probs)
        if model is not None:
            return model
        warnings.warn(
            "hmmlearn not available or incompatible. Using native FiberHMM implementation."
        )

    # Use native implementation
    model = FiberHMM(n_states=2)
    model.emissionprob_ = emission_probs
    return model


def _try_create_hmmlearn(emission_probs: np.ndarray):
    """Try to create an hmmlearn model, handling version differences."""
    try:
        import hmmlearn
        version = getattr(hmmlearn, '__version__', '0.0.0')
        major_version = int(version.split('.')[0]) if version else 0
        minor_version = int(version.split('.')[1]) if len(version.split('.')) > 1 else 0

        # hmmlearn >= 0.3 uses CategoricalHMM
        if major_version > 0 or (major_version == 0 and minor_version >= 3):
            from hmmlearn.hmm import CategoricalHMM

            model = CategoricalHMM(
                n_components=2,
                init_params='',  # Don't auto-initialize
                params='st',     # Train start and transition probs only
                n_iter=1000
            )
            model.emissionprob_ = emission_probs
            return model

        else:
            # Old hmmlearn with MultinomialHMM
            from hmmlearn.hmm import MultinomialHMM

            model = MultinomialHMM(
                n_components=2,
                init_params='',
                params='st',
                n_iter=1000
            )
            model.emissionprob_ = emission_probs
            return model

    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"hmmlearn initialization failed: {e}")
        return None


def train_model(emission_probs: np.ndarray,
                train_data: np.ndarray,
                n_iterations: int = 10,
                use_legacy: bool = False,
                normalize: bool = True) -> Tuple[FiberHMM, list]:
    """
    Train multiple HMM models and return the best one.

    Args:
        emission_probs: Fixed emission probabilities
        train_data: Dictionary mapping iteration -> observation array
        n_iterations: Number of random initializations
        use_legacy: Try to use hmmlearn for training
        normalize: If True, normalize states so State 0 = accessible (default True)

    Returns:
        (best_model, all_models)
    """
    from fiberhmm.core.model_io import _convert_hmmlearn_model

    best_model = None
    best_logprob = float('-inf')
    all_models = []

    pbar = tqdm(range(n_iterations), desc="Training iterations",
                disable=not HAS_TQDM)

    for i in pbar:
        np.random.seed(i)

        # Random initialization
        start_probs = np.random.dirichlet((1, 1))
        transition_probs = np.random.dirichlet((1, 1), 2)

        # Create model
        if use_legacy:
            model = _try_create_hmmlearn(emission_probs)
            if model is None:
                model = FiberHMM(n_states=2)
                model.emissionprob_ = emission_probs
        else:
            model = FiberHMM(n_states=2)
            model.emissionprob_ = emission_probs

        model.startprob_ = start_probs
        model.transmat_ = transition_probs

        # Train
        if isinstance(train_data, dict):
            data = train_data[i % len(train_data)]
        else:
            data = train_data

        training = data.reshape(-1, 1)
        model.fit(training, lengths=[len(data)], verbose=True, desc=f"Init {i+1} EM")

        # Get log probability
        if hasattr(model, 'monitor_') and model.monitor_ and model.monitor_.history:
            logprob = model.monitor_.history[-1]
        else:
            logprob = model.score(training)

        # Convert to native if needed
        if not isinstance(model, FiberHMM):
            model = _convert_hmmlearn_model(model)

        all_models.append(model)

        if logprob > best_logprob:
            best_logprob = logprob
            best_model = model

        # Update progress bar with current best
        if HAS_TQDM:
            pbar.set_postfix({'best_logprob': f'{best_logprob:.2e}'})

    # Normalize states for best model and all models
    if normalize:
        if best_model is not None:
            best_model.normalize_states()
        for model in all_models:
            model.normalize_states()

    return best_model, all_models


# =============================================================================
# Backward-compatible re-exports from model_io
# =============================================================================

try:
    from fiberhmm.core.model_io import (  # noqa: F401
        load_model,
        save_model,
        load_model_with_metadata,
    )
except ImportError:
    # Allow hmm.py to be imported standalone (e.g., from top-level scripts
    # that add the directory to sys.path) without model_io being available
    # as a package import. In that case, callers must import model_io directly.
    pass
