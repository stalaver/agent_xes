"""
Bi-LSTM + SMOTE Baseline - Bidirectional LSTM binary classifier with oversampling

Purpose: Train a bidirectional LSTM to directly classify symbolized prefixes as
success or failure.  SMOTE is applied to the padded integer sequences before
training to compensate for class imbalance.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.base import BaseBaseline
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM binary classifier."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return failure probability for each sample.

        Args:
            x: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Tensor of shape ``(batch,)`` with values in [0, 1].
        """
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # Concatenate final forward and backward hidden states
        fwd = h_n[-2]
        bwd = h_n[-1]
        hidden = torch.cat([fwd, bwd], dim=-1)
        return torch.sigmoid(self.fc(hidden)).squeeze(-1)


class BiLSTMBaseline(BaseBaseline):
    """Bi-LSTM baseline with SMOTE oversampling for class balance.

    Uses a bidirectional LSTM to classify symbolized prefixes as success
    or failure, with SMOTE applied to the padded integer sequences before
    training to address the typical class imbalance in agent traces.
    """

    name: str = "bilstm"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 30,
        lr: float = 0.001,
        device: str = "cpu",
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)

        self.vocab_: Optional[dict[str, int]] = None
        self.model_: Optional[BiLSTMModel] = None
        self.max_length_: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Build vocabulary, apply SMOTE, and train the Bi-LSTM.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        self.vocab_ = self._build_vocab(entries)
        vocab_size = len(self.vocab_)
        logger.info("BiLSTM vocabulary built: %d symbols", vocab_size)

        X_raw, labels = self._encode_and_pad(entries)
        self.max_length_ = X_raw.shape[1]

        X_train, y_train = self._apply_smote(X_raw, labels)

        dataset = TensorDataset(
            torch.from_numpy(X_train).long(),
            torch.from_numpy(y_train).float(),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model = BiLSTMModel(
            vocab_size, self.hidden_size, self.num_layers,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
            if epoch % 5 == 0 or epoch == 1:
                logger.info("BiLSTM epoch %d/%d  loss=%.4f", epoch, self.epochs, total_loss / len(dataset))

        model.eval()
        self.model_ = model
        logger.info("BiLSTM training complete on %d samples (after SMOTE)", len(dataset))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, symbols: list[str]) -> float:
        """Return failure probability for a symbol sequence.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            Failure probability in [0.0, 1.0].

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.vocab_ is None or self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if not symbols:
            return 0.0

        ids = self._encode(symbols)
        padded = self._pad_sequence(ids, self.max_length_)
        x = torch.tensor([padded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            prob = self.model_(x)
        return float(prob.item())

    def predict_at_k(self, symbols: list[str], k: int) -> float:
        """Return failure probability using only the first *k* symbols.

        Args:
            symbols: Full symbolized prefix sequence.
            k: Number of leading symbols to use.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        return self.predict(symbols[:k])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vocab(self, entries: list[PrefixEntry]) -> dict[str, int]:
        """Create symbol-to-index mapping from all entries."""
        symbols = sorted({s for e in entries for s in e.symbols})
        vocab: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        for sym in symbols:
            if sym not in vocab:
                vocab[sym] = len(vocab)
        return vocab

    def _encode(self, symbols: list[str]) -> list[int]:
        """Encode a symbol sequence to integer ids."""
        assert self.vocab_ is not None
        return [self.vocab_.get(s, UNK_IDX) for s in symbols]

    @staticmethod
    def _pad_sequence(ids: list[int], max_len: int) -> list[int]:
        """Right-pad or truncate an id list to *max_len*."""
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [PAD_IDX] * (max_len - len(ids))

    def _encode_and_pad(
        self, entries: list[PrefixEntry],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode and pad all entries into a 2-D integer array plus labels.

        Returns:
            Tuple of ``(X, y)`` where *X* has shape ``(n, max_len)``
            and *y* has shape ``(n,)``.
        """
        assert self.vocab_ is not None
        encoded = [self._encode(e.symbols) for e in entries]
        max_len = max((len(seq) for seq in encoded), default=1)

        X = np.full((len(encoded), max_len), PAD_IDX, dtype=np.int64)
        for i, seq in enumerate(encoded):
            X[i, : len(seq)] = seq

        y = self._labels_from_entries(entries).astype(np.int64)
        return X, y

    @staticmethod
    def _apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Balance classes with SMOTE, rounding back to valid integer indices.

        SMOTE interpolates over the padded integer array which can produce
        fractional symbol indices.  We round the result so the values remain
        valid embedding indices and clamp negatives to ``PAD_IDX``.

        Returns:
            Tuple of ``(X_resampled, y_resampled)`` with integer *X*.
        """
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning("Only one class present; skipping SMOTE.")
            return X, y

        min_count = int(np.min(np.bincount(y)))
        k_neighbors = min(5, min_count - 1)
        if k_neighbors < 1:
            logger.warning("Minority class too small for SMOTE (n=%d); skipping.", min_count)
            return X, y

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)
        X_res = np.round(X_res).astype(np.int64)
        X_res = np.clip(X_res, 0, None)
        logger.info(
            "SMOTE resampling: %d -> %d samples (class distribution: %s)",
            len(y),
            len(y_res),
            dict(zip(*np.unique(y_res, return_counts=True))),
        )
        return X_res, y_res
