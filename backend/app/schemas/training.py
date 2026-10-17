from typing import Literal
from pydantic import BaseModel, Field

TrainModel = Literal['yolo26n', 'yolo26s', 'rtdetr_l', 'fasterrcnn_mobilenet_v3']


class TrainingRequest(BaseModel):
    model_name: TrainModel = 'yolo26n'
    epochs: int = Field(30, ge=1, le=300)
    device: Literal['auto', 'cpu', 'gpu'] = 'auto'
    image_size: int = Field(640, ge=320, le=1280, multiple_of=32)
    batch_size: int = Field(1, ge=1, le=16)
    validation_fraction: float = Field(.2, ge=.1, le=.4, allow_inf_nan=False)
    seed: int = Field(42, ge=0, le=2**32-1)
    labels_reviewed: Literal[True]
    include_verified_empty: bool = False
