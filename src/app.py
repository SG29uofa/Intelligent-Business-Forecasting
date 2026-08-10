from flask import Flask, render_template

from forecast import (
    get_forecast_summary,
    get_forecast_chart_data
)

from ollama_explainer import explain_forecast


app = Flask(__name__)


@app.route("/")
def home():

    summary = get_forecast_summary()

    chart_data = get_forecast_chart_data()

    ai_explanation = explain_forecast(summary)

    return render_template(
        "index.html",
        summary=summary,
        chart_data=chart_data,
        ai_explanation=ai_explanation
    )


if __name__ == "__main__":
    app.run(debug=True)