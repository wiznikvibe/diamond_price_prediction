import os, sys 
from src.exception import CustomException
from src.logger import logging
from src.entity import artifact_entity
from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    try:
        if request.method == "GET":
            return render_template('home.html')
    except Exception as e:
        logging.info(f"{CustomException(e,sys)}")
        raise CustomException(e,sys)    
    
    # else:
    #     data = CustomData(
    #         carat= float(request.form.get('carat')),
    #         cut = request.form.get('cut'),
    #         color = request.form.get("color"),
    #         clarity = request.form.get('clarity'),
    #         x = float(request.form.get('X')),
    #         y = float(request.form.get('Y')),
    #         z = float(request.form.get('Z')),
    #     )
    # final_data = data.get_as_dataframe()
    # pipeline = PredictionPipeline()
    # prediction = pipeline.predict(final_data)

    # return render_template("result.html", final_result = round(prediction[0], 2))   

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('form.html')
        else:
        
            data = CustomData(
                carat= float(request.form.get('carat')),
                cut = request.form.get('cut'),
                color = request.form.get("color"),
                clarity = request.form.get('clarity'),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
            )
        final_data = data.get_as_dataframe()
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(final_data)

        return render_template("result.html", final_result = round(prediction[0], 2))   
    except Exception as e:
        logging.info(f"{CustomException(e,sys)}")
        raise CustomException(e,sys) 

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)
















































# prediction = PredictionPipeline()

# data = CustomData(
#     carat=0.33,
#     cut='Premium',
#     color='E',
#     clarity='VS2',
#     x=4.39,
#     y=4.43,
#     z=2.72
# )

# final_data = data.get_as_dataframe()
# print(final_data)
# pred = prediction.predict(final_data)
# print(round(pred[0], 2))


