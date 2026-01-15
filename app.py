from flask import Flask, render_template, request
from src.pipelines.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Product ID": request.form["product_id"],
            "Type": request.form["type"],
            "Air temperature [K]": float(request.form["air_temp"]),
            "Process temperature [K]": float(request.form["process_temp"]),
            "Rotational speed [rpm]": float(request.form["rpm"]),
            "Torque [Nm]": float(request.form["torque"]),
            "Tool wear [min]": float(request.form["tool_wear"])
        }

        pipeline = PredictionPipeline()
        result = pipeline.predict(data)

        return render_template(
            "index.html",
            prediction=result["prediction"],
            probability=result["failure_probability"]
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Something went wrong. Please check inputs."
        )

if __name__ == "__main__":
    app.run(debug=True)
