"""NeurIPS Paper Checklist and Rigor Rubric Analyzer using Zero-Shot NLI."""

import logging
from typing import Any
import pandas as pd
import torch
from transformers import pipeline

from litreview.analyzers.base import Analyzer

logger = logging.getLogger(__name__)

NEURIPS_RUBRIC_CRITERIA = {
    "limitations": "discusses limitations assumptions scope and potential failure modes",
    "reproducibility": "provides open source code repository dataset artifacts or reproduction commands",
    "experimental_rigor": "evaluates baselines ablation studies benchmarks and quantitative metrics",
    "statistical_significance": "reports confidence intervals error bars variance or statistical hypothesis tests",
    "compute_resources": "specifies compute resources hardware accelerators training time or gpu hours",
}


class NeurIPSRubricAnalyzer(Analyzer):
    """Evaluates paper abstracts and texts against NeurIPS checklist criteria.

    Uses zero-shot NLI entailment probabilities to quantify the methodological
    rigor and transparency of each study.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._pipeline = None
        self._results: pd.DataFrame | None = None

    def fit(self, texts: pd.Series) -> "NeurIPSRubricAnalyzer":
        """Initialize the classification pipeline on GPU or CPU."""
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
            model=self.model_name,
            device=device,
            dtype=dtype,
        )
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Score each text against the NeurIPS rubric criteria.

        Returns:
            DataFrame with columns for each criterion score and an aggregated 'rubric_score'.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before transform()")

        candidate_labels = list(NEURIPS_RUBRIC_CRITERIA.values())
        criterion_names = list(NEURIPS_RUBRIC_CRITERIA.keys())

        empty_mask = texts.apply(lambda x: pd.isna(x) or not str(x).strip())
        empty_indices = set(texts[empty_mask].index.tolist())
        non_empty = texts[~empty_mask].tolist()

        scored_records = []
        if non_empty:
            pipeline_outputs = self._pipeline(
                non_empty,
                candidate_labels=candidate_labels,
                multi_label=True,
                batch_size=32,
                truncation=True,
                max_length=256,
            )

            # Ensure pipeline_outputs is iterable over results
            if isinstance(pipeline_outputs, dict):
                pipeline_outputs = [pipeline_outputs]

            for out in pipeline_outputs:
                label_to_score = dict(zip(out["labels"], out["scores"]))
                record = {}
                for name, label in zip(criterion_names, candidate_labels):
                    record[f"rubric_{name}"] = round(label_to_score.get(label, 0.0), 4)

                # Composite rigor score (mean of criteria)
                record["rubric_score"] = round(sum(record.values()) / len(record), 4)
                scored_records.append(record)

        # Re-align with original indices
        rec_iter = iter(scored_records)
        final_rows = []
        for idx in texts.index:
            if idx in empty_indices:
                empty_row = {f"rubric_{name}": 0.0 for name in criterion_names}
                empty_row["rubric_score"] = 0.0
                final_rows.append(empty_row)
            else:
                final_rows.append(next(rec_iter))

        df = pd.DataFrame(final_rows, index=texts.index)
        self._results = df
        return df

    @property
    def results(self) -> dict[str, Any]:
        """Summary statistics across the analyzed corpus."""
        if self._results is None:
            return {}

        means = {col: round(float(self._results[col].mean()), 4) for col in self._results.columns}
        return {
            "criteria_means": means,
            "overall_mean_rigor": means.get("rubric_score", 0.0),
        }
