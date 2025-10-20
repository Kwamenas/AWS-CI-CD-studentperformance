from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)


@application.route('/')
def index():
    return render_template("index.html")


@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score',0)),
            writing_score=float(request.form.get('writing_score',0))

        )

        df=data.trans_dataframe()
        print(df)
        print("Before prediction")

        predic_pipeline=PredictPipeline()
        print(" Called the predict pipe")
        results=predic_pipeline.predict(df)

        print("Data predicted")

        return render_template("home.html",results=results[0])

if __name__=="__main__":
    application.run(host="0.0.0.0",port=5000,debug=True)  
