"""Zero-shot classification analyzer for paper topic validation."""

import pandas as pd
import torch
from tqdm import tqdm

from litreview.analyzers.base import Analyzer
from litreview.config import ZeroShotConfig


class ZeroShotAnalyzer(Analyzer):
    """Classify papers into predefined topics using zero-shot classification.

    Uses HuggingFace transformers pipeline for multi-label zero-shot
    classification. Supports multiple models and configurable thresholds.
    """

    def __init__(self, config: ZeroShotConfig | None = None, **kwargs):
        if config is None:
            config = ZeroShotConfig(**kwargs)
        self.config = config
        self._pipeline = None
        self._results: pd.DataFrame | None = None

    @property
    def models(self) -> list[str]:
        return self.config.models

    @property
    def threshold(self) -> float:
        return self.config.threshold

    def fit(self, texts: pd.Series) -> "ZeroShotAnalyzer":
        """Initialize zero-shot classification pipeline.

        Args:
            texts: Series of text strings to classify.

        Returns:
            self for chaining.
        """
        from transformers import pipeline

        # Robust GPU detection: fall back to device 0 if CUDA is present
        if torch.cuda.is_available():
            device = 0
            use_bf16 = torch.cuda.is_bf16_supported()
        elif torch.cuda.device_count() > 0:
            device = 0
            use_bf16 = False
        else:
            device = -1
            use_bf16 = False

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.config.models[0],
            device=device,
            dtype=dtype,
        )
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Classify texts into candidate labels.

        Must call fit() first.

        Args:
            texts: Series of text strings to classify.

        Returns:
            DataFrame with 'label', 'score', and 'classified' columns.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before transform()")

        labels = list(self.config.candidate_labels.values())
        if not labels:
            raise ValueError(
                "No candidate labels configured. "
                "Add labels to config.yaml under zeroshot.candidate_labels."
            )

        empty_mask = texts.apply(lambda x: pd.isna(x) or not str(x).strip())
        empty_indices = set(texts[empty_mask].index.tolist())
        non_empty = texts[~empty_mask].tolist()

        results = []
        if non_empty:
            batch_size = getattr(self.config, "batch_size", 64)

            # Generator yielding individual strings
            def item_generator():
                for text in non_empty:
                    yield text

            # Stream individual strings into pipeline; pass batch_size for GPU batching
            pipeline_outputs = self._pipeline(
                item_generator(),
                candidate_labels=labels,
                multi_label=True,
                batch_size=batch_size,
                truncation=True,
                max_length=256,
            )

            # tqdm steps for every single text yielded back
            for item in tqdm(
                pipeline_outputs,
                total=len(non_empty),
                desc="Zero-shot classification",
            ):
                scores = item["scores"]
                best_idx = scores.index(max(scores))
                results.append({
                    "label": item["labels"][best_idx],
                    "score": scores[best_idx],
                })

        # Interleave results back, preserving original order
        result_iter = iter(results)
        final = []
        for idx in texts.index:
            if idx in empty_indices:
                final.append({"label": "unknown", "score": 0.0})
            else:
                final.append(next(result_iter))

        df = pd.DataFrame(final, index=texts.index)
        df["classified"] = df["score"] >= self.config.threshold
        self._results = df
        return df

    @property
    def results(self) -> dict:
        """Return analysis results as dict.

        Returns:
            Dict with classifications DataFrame, label_counts (all labels, 0 for missing), and threshold.
        """
        if self._results is None:
            # Return all labels with 0 counts even before classification runs
            label_counts = {lbl: 0 for lbl in self.config.candidate_labels.values()}
            return {"label_counts": label_counts, "threshold": self.config.threshold}

        # Start with all configured labels at 0, then overlay actual counts
        label_counts = {lbl: 0 for lbl in self.config.candidate_labels.values()}
        actual_counts = self._results["label"].value_counts().to_dict()
        label_counts.update(actual_counts)

        return {
            "classifications": self._results,
            "label_counts": label_counts,
            "threshold": self.config.threshold,
        }