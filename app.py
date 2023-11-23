import os, sys
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData


prediction = PredictionPipeline()

data = CustomData(
    carat=0.33,
    cut='Premium',
    color='E',
    clarity='VS2',
    x=4.39,
    y=4.43,
    z=2.72
)

final_data = data.get_as_dataframe()
print(final_data)
pred = prediction.predict(final_data)
print(round(pred[0], 2))


