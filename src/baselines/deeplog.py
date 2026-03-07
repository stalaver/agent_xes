"""
DeepLog Baseline - LSTM next-symbol prediction for anomaly detection

Purpose: Train an LSTM on successful traces to learn "normal" symbol
sequences, then score new traces by how often the actual next symbol
falls outside the model's top-k predictions.

Reference: Du et al., "DeepLog: Anomaly Detection and Diagnosis from System
Logs through Deep Learning", CCS 2017.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.base import BaseBaseline
from src.data_collection.trace_schema import TaskOutcome
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


class DeepLogModel(nn.Module):
    """LSTM that predicts the next symbol given a prefix."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return next-symbol logits for every position.

        Args:
            x: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out)


class DeepLogBaseline(BaseBaseline):
    """DeepLog anomaly detector based on next-symbol prediction.

    Trains only on successful traces so the LSTM learns normal behaviour.
    At inference time, a high fraction of mispredicted next-symbols signals
    an anomalous (likely failing) trace.
    """

    name: str = "deeplog"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 20,
        lr: float = 0.001,
        top_k: int = 3,
        device: str = "cpu",
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.top_k = top_k
        self.device = torch.device(device)

        self.vocab_: Optional[dict[str, int]] = None
        self.model_: Optional[DeepLogModel] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Build vocabulary from all entries and train on success-only traces.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        self.vocab_ = self._build_vocab(entries)
        vocab_size = len(self.vocab_)
        logger.info("DeepLog vocabulary built: %d symbols", vocab_size)

        success_seqs = [
            e.symbols for e in entries if e.outcome == TaskOutcome.SUCCESS
        ]
        if not success_seqs:
            logger.warning("No successful traces for DeepLog training; model will predict 1.0 for everything.")
            return

        input_ids, target_ids = self._make_training_pairs(success_seqs)
        if input_ids.size(0) == 0:
            logger.warning("All success sequences have length <= 1; nothing to train on.")
            return

        dataset = TensorDataset(input_ids, target_ids)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model = DeepLogModel(vocab_size, self.hidden_size, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = model(x_batch)
                loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
            if epoch % 5 == 0 or epoch == 1:
                logger.info("DeepLog epoch %d/%d  loss=%.4f", epoch, self.epochs, total_loss / len(dataset))

        model.eval()
        self.model_ = model
        logger.info("DeepLog training complete on %d success sequences", len(success_seqs))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, symbols: list[str]) -> float:
        """Return failure probability based on next-symbol misprediction rate.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            Failure probability in [0.0, 1.0].

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.vocab_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.model_ is None:
            return 1.0
        if len(symbols) <= 1:
            return 0.0

        ids = self._encode(symbols)
        input_t = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        target_ids = ids[1:]

        with torch.no_grad():
            logits = self.model_(input_t)
        topk = torch.topk(logits[0], self.top_k, dim=-1).indices.cpu().numpy()

        misses = sum(
            1 for t, pred_row in zip(target_ids, topk) if t not in pred_row
        )
        return misses / len(target_ids)

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

    def _make_training_pairs(
        self, sequences: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build padded input/target tensors for next-symbol prediction.

        Each sequence ``[s0, s1, s2, s3]`` produces input ``[s0, s1, s2]``
        and target ``[s1, s2, s3]``.  Sequences of length <= 1 are skipped.
        Shorter pairs are right-padded with ``PAD_IDX``.

        Returns:
            Tuple of ``(input_ids, target_ids)`` tensors.
        """
        assert self.vocab_ is not None
        pairs: list[tuple[list[int], list[int]]] = []
        for seq in sequences:
            if len(seq) <= 1:
                continue
            ids = self._encode(seq)
            pairs.append((ids[:-1], ids[1:]))

        if not pairs:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        max_len = max(len(inp) for inp, _ in pairs)
        input_arr = np.full((len(pairs), max_len), PAD_IDX, dtype=np.int64)
        target_arr = np.full((len(pairs), max_len), PAD_IDX, dtype=np.int64)
        for i, (inp, tgt) in enumerate(pairs):
            input_arr[i, : len(inp)] = inp
            target_arr[i, : len(tgt)] = tgt

        return torch.from_numpy(input_arr), torch.from_numpy(target_arr)
