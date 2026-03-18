from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClientType(str, Enum):
    VIP = "VIP"
    REGULAR = "regular"


class ChannelType(str, Enum):
    WEB = "web"
    PHONE = "phone"
    BRANCH = "branch"
    EMAIL = "email"
    CHAT = "chat"


class PriorityLabel(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class CRMFeatures(BaseModel):
    amount: float = Field(ge=0.0)
    repeat_count: int = Field(ge=0)
    client_type: ClientType
    account_age: int = Field(ge=0, description="Account age in days.")
    channel: ChannelType


class ComplaintRequest(BaseModel):
    complaint_id: str
    narrative: str = Field(min_length=1)
    crm: CRMFeatures


class CategoryOutput(BaseModel):
    label: str
    proba: dict[str, float]


class LegalOutput(BaseModel):
    proba_legal_threat: float = Field(ge=0.0, le=1.0)
    is_legal_threat: bool


class PriorityOutput(BaseModel):
    label: PriorityLabel
    proba: dict[PriorityLabel, float]
    threshold_p1: float = Field(ge=0.0, le=1.0)


class InferenceOutput(BaseModel):
    complaint_id: str
    category: CategoryOutput
    legal: LegalOutput
    priority: PriorityOutput
    metadata: dict[str, Any] = Field(default_factory=dict)

