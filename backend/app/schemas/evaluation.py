import math
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from app.schemas.qc import QCConfig


class GoldenBox(BaseModel):
    label: str = Field(min_length=1)
    geometry: dict[str, float]

    @model_validator(mode='after')
    def valid_box(self):
        g = self.geometry
        if set(g) != {'x1', 'y1', 'x2', 'y2'} or any(not math.isfinite(v) or v < 0 for v in g.values()):
            raise ValueError('Box must have four finite nonnegative pixel coordinates')
        if g['x2'] <= g['x1'] or g['y2'] <= g['y1'] or not self.label.strip():
            raise ValueError('Box must have positive area and a nonempty label')
        return self


class GoldenFrame(BaseModel):
    file_name: str = Field(min_length=1)
    annotations: list[GoldenBox]


class BenchmarkRequest(BaseModel):
    verified_by: str = Field(min_length=1)
    notes: str = ''
    includes_unwarned_regions: Literal[True]
    samples: list[GoldenFrame] = Field(min_length=1)


class EvaluateRequest(BaseModel):
    benchmark_id: int
    baseline_model: str = Field(min_length=1)
    pilot_models: list[str] = Field(min_length=2)
    config: QCConfig = Field(default_factory=QCConfig)
    evaluation_iou: float = Field(.5, ge=.1, le=.95)
    review_minutes: dict[Literal['baseline', 'multi_model'], float] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_comparison(self):
        if len(set(self.pilot_models)) != len(self.pilot_models):
            raise ValueError('Pilot models must be distinct')
        if self.baseline_model not in self.pilot_models:
            raise ValueError('Pilot must include the baseline model')
        if any(not math.isfinite(v) or v <= 0 for v in self.review_minutes.values()):
            raise ValueError('Measured review minutes must be finite and positive')
        return self


class AcceptanceRequest(BaseModel):
    status: Literal['SUBMITTED', 'RETURNED', 'ACCEPTED']
    strategy: Literal['baseline', 'multi_model']
    actor: str = Field(min_length=1)
    reason: str = ''

    @model_validator(mode='after')
    def validate_reason(self):
        if not self.actor.strip() or (self.status == 'RETURNED' and not self.reason.strip()):
            raise ValueError('Actor and return reason are required')
        return self
