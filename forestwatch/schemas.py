from pydantic import BaseModel, Field

class ChangeRequest(BaseModel):
    earlier_date: str
    later_date: str

class PredictionRequest(BaseModel):
    model: str = Field(default="Linear Regression", pattern="^(Linear Regression|Random Forest)$")

class Credentials(BaseModel):
    email: str
    password: str
