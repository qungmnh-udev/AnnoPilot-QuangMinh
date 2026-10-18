from typing import Literal
from pydantic import BaseModel, Field, model_validator


class WBFExperimentRequest(BaseModel):
    baseline_model: Literal['yolo26n', 'yolo26s'] = 'yolo26n'
    models: list[str] = Field(default_factory=lambda: ['yolo26n', 'fasterrcnn_mobilenet_v3', 'rtdetr_l'])
    sample_limit: int = Field(5, ge=1, le=30)
    repeats: int = Field(4, ge=2, le=8, description='Total views including the original; default covers all three perturbations')
    seed: int = Field(42, ge=0, le=2**32 - 1)
    gaussian_sigma: float = Field(8, ge=0, le=30, allow_inf_nan=False)
    brightness_delta: float = Field(.15, ge=0, le=.5, allow_inf_nan=False)
    blur_radius: float = Field(.7, ge=0, le=3, allow_inf_nan=False)
    confidence: float = Field(.35, ge=.05, le=.95, allow_inf_nan=False)
    fusion_iou: float = Field(.55, ge=.1, le=.95, allow_inf_nan=False)
    benchmark_id: int | None = Field(None, gt=0)

    @model_validator(mode='after')
    def valid_models(self):
        allowed = {self.baseline_model, 'fasterrcnn_mobilenet_v3', 'rtdetr_l'}
        if (len(self.models) < 2 or len(set(self.models)) != len(self.models)
                or any(m not in allowed for m in self.models) or self.baseline_model not in self.models):
            raise ValueError('Select the baseline YOLO and at least one distinct supported detector')
        return self
