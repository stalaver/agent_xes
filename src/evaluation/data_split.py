"""
Data Splitter - Train/val/test splitting with stratification and holdout sites

Purpose: Provide reproducible, stratified dataset splits for evaluation.
Supports website-level holdout for cross-site generalization testing.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split

from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Result of splitting a prefix dataset into train/val/test/holdout."""

    train: list[PrefixEntry] = field(default_factory=list)
    val: list[PrefixEntry] = field(default_factory=list)
    test: list[PrefixEntry] = field(default_factory=list)
    holdout: list[PrefixEntry] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        """Return split sizes.

        Returns:
            Dict mapping split name to entry count.
        """
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "holdout": len(self.holdout),
        }


class DataSplitter:
    """Handles train/val/test splitting with stratification and holdout sites."""

    def split(
        self,
        entries: list[PrefixEntry],
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        holdout_sites: list[str] | None = None,
        seed: int = 42,
    ) -> DataSplit:
        """Split entries into train/val/test with optional website holdout.

        Stratifies by outcome and website when possible, falling back to
        outcome-only stratification when any composite stratum has fewer
        than 2 samples.

        Args:
            entries: Full list of prefix entries to split.
            test_ratio: Fraction of non-holdout data reserved for testing.
            val_ratio: Fraction of non-holdout data reserved for validation.
            holdout_sites: Websites to remove entirely before splitting.
            seed: Random seed for reproducibility.

        Returns:
            DataSplit with train, val, test, and holdout lists.
        """
        if not entries:
            logger.warning("Empty entry list — returning empty DataSplit")
            return DataSplit()

        holdout: list[PrefixEntry] = []
        remaining: list[PrefixEntry] = entries

        if holdout_sites:
            site_set = set(holdout_sites)
            holdout = [e for e in entries if e.website in site_set]
            remaining = [e for e in entries if e.website not in site_set]
            logger.info(
                "Holdout sites %s: %d entries held out, %d remaining",
                holdout_sites,
                len(holdout),
                len(remaining),
            )

        if len(remaining) < 3:
            logger.warning(
                "Only %d entries after holdout — all go to train", len(remaining)
            )
            return DataSplit(train=remaining, holdout=holdout)

        strat_keys = self._stratification_keys(remaining)

        test_size = max(1, int(len(remaining) * test_ratio))
        train_val, test_split = self._safe_split(
            remaining, strat_keys, test_size=test_size, seed=seed
        )

        if len(train_val) < 2 or val_ratio <= 0.0:
            return DataSplit(
                train=train_val, val=[], test=test_split, holdout=holdout
            )

        val_fraction_of_remainder = val_ratio / (1.0 - test_ratio)
        val_size = max(1, int(len(train_val) * val_fraction_of_remainder))
        strat_keys_tv = self._stratification_keys(train_val)
        train_split, val_split = self._safe_split(
            train_val, strat_keys_tv, test_size=val_size, seed=seed
        )

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d, holdout: %d",
            len(train_split),
            len(val_split),
            len(test_split),
            len(holdout),
        )

        return DataSplit(
            train=train_split,
            val=val_split,
            test=test_split,
            holdout=holdout,
        )

    @staticmethod
    def _stratification_keys(entries: list[PrefixEntry]) -> list[str]:
        """Build stratification keys, falling back if strata are too small.

        Tries composite ``outcome_website`` keys first. If any stratum has
        fewer than 2 members, falls back to outcome-only keys.

        Args:
            entries: Entries to compute keys for.

        Returns:
            List of stratification key strings aligned with *entries*.
        """
        composite = [f"{e.outcome.value}_{e.website}" for e in entries]
        counts = Counter(composite)
        if all(c >= 2 for c in counts.values()):
            return composite

        logger.debug(
            "Composite strata too small (min=%d) — falling back to outcome-only",
            min(counts.values()),
        )
        return [e.outcome.value for e in entries]

    @staticmethod
    def _safe_split(
        entries: list[PrefixEntry],
        strat_keys: list[str],
        test_size: int,
        seed: int,
    ) -> tuple[list[PrefixEntry], list[PrefixEntry]]:
        """Wrapper around train_test_split that falls back to unstratified.

        Args:
            entries: Data to split.
            strat_keys: Stratification labels.
            test_size: Number of entries for the second split.
            seed: Random seed.

        Returns:
            Tuple of (first_split, second_split).
        """
        try:
            a, b = train_test_split(
                entries,
                test_size=test_size,
                random_state=seed,
                stratify=strat_keys,
            )
        except ValueError:
            logger.warning("Stratified split failed — falling back to random split")
            a, b = train_test_split(
                entries, test_size=test_size, random_state=seed
            )
        return a, b
