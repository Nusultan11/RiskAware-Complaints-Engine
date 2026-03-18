from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_aware.domain.schemas import (
    CategoryOutput,
    ComplaintRequest,
    InferenceOutput,
    LegalOutput,
    PriorityLabel,
    PriorityOutput,
)
from risk_aware.features.assembler import FeatureAssembler
from risk_aware.models.base import TabularClassifier, TextClassifier
from risk_aware.models.thresholds import ThresholdPolicy


@dataclass(slots=True)
class InferenceOrchestrator:
    category_model: TextClassifier
    legal_model: TextClassifier
    priority_model: TabularClassifier
    threshold_policy: ThresholdPolicy

    def __post_init__(self) -> None:
        self.assembler = FeatureAssembler(category_labels=self.category_model.labels)

    def predict(self, request: ComplaintRequest) -> InferenceOutput:
        texts = [request.narrative]

        category_vector = self.category_model.predict_proba(texts)[0]
        legal_vector = self.legal_model.predict_proba(texts)[0]
        legal_proba = float(legal_vector[-1])

        priority_features = self.assembler.build_priority_vector(category_vector, legal_proba, request.crm)
        priority_proba = self.priority_model.predict_proba(np.expand_dims(priority_features, axis=0))[0]
        p1, p2, p3 = float(priority_proba[0]), float(priority_proba[1]), float(priority_proba[2])

        priority_label = self.threshold_policy.resolve_priority(p1=p1, p2=p2, p3=p3)

        category_idx = int(np.argmax(category_vector))
        category_label = self.category_model.labels[category_idx]
        category_map = {
            self.category_model.labels[i]: float(category_vector[i]) for i in range(len(self.category_model.labels))
        }

        return InferenceOutput(
            complaint_id=request.complaint_id,
            category=CategoryOutput(label=category_label, proba=category_map),
            legal=LegalOutput(proba_legal_threat=legal_proba, is_legal_threat=legal_proba >= 0.5),
            priority=PriorityOutput(
                label=PriorityLabel(priority_label),
                proba={
                    PriorityLabel.P1: p1,
                    PriorityLabel.P2: p2,
                    PriorityLabel.P3: p3,
                },
                threshold_p1=self.threshold_policy.threshold_p1,
            ),
            metadata={"decision_mode": self.threshold_policy.mode},
        )

