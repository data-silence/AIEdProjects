from pydantic import BaseModel

class PredictionResult(BaseModel):
    text: str
    prediction: str