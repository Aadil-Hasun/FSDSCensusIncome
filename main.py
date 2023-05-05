from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            age=float(request.form.get('age')),
            workclass=request.form.get('workclass'),
            fnlwgt=float(request.form.get('fnlwgt')),
            education_num=int(request.form.get('education-num')),
            marital_status=request.form.get('marital-status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            capital_gain=float(request.form.get('capital-gain')),
            capital_loss=float(request.form.get('capital-loss')),
            hours_per_week=float(request.form.get('hours-per-week')),
            native_country=request.form.get('native-country'),
            Gender=int(request.form.get('Gender'))
        )
        data_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred_result = predict_pipeline.predict(data_df)

        return render_template('result.html', result=pred_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
