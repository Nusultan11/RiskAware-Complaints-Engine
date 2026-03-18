from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_aware.domain.schemas import CRMFeatures


def encode_client_type(value: str) -> float:
    return 1.0 if value.upper() == "VIP" else 0.0


def encode_channel(value: str) -> float:
    mapping = {"web": 0.0, "phone": 1.0, "branch": 2.0, "email": 3.0, "chat": 4.0}
    return mapping.get(value.lower(), 0.0)


@dataclass(slots=True)
class FeatureAssembler:
    category_labels: list[str]

    def build_priority_vector(
        self,
        category_proba: np.ndarray,
        legal_proba: float,
        crm: CRMFeatures,
    ) -> np.ndarray:
        vector: list[float] = []
        vector.extend(category_proba.tolist())
        vector.append(float(legal_proba))
        vector.append(float(crm.amount))
        vector.append(float(crm.repeat_count))
        vector.append(encode_client_type(crm.client_type.value))
        vector.append(float(crm.account_age))
        vector.append(encode_channel(crm.channel.value))
        return np.asarray(vector, dtype=float)

