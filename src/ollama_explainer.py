import subprocess
import re


def build_fallback_explanation(summary):

    return (
        f"The {summary['forecast_days']}-day forecast projects average sales of "
        f"{summary['average_forecast']:,.0f}. Forecasted sales range from "
        f"{summary['lowest_sales']:,.0f} on {summary['lowest_date']} to "
        f"{summary['peak_sales']:,.0f} on {summary['peak_date']}."
        "\n\n"
        "The variation between the highest and lowest forecast periods "
        "may warrant additional business review."
    )


def is_valid_explanation(output, summary):

    lower_output = output.lower()

    forbidden_terms = [
        "demand",
        "inventory",
        "staffing",
        "revenue",
        "profit",
        "unit",
        "units",
        "approximately",
        "specific date",
        "expect to",
        "expected to"
    ]

    for term in forbidden_terms:
        if term in lower_output:
            return False

    # Require all important supplied facts to appear
    required_values = [
        str(summary["forecast_days"]),
        str(round(summary["average_forecast"])),
        str(round(summary["peak_sales"])),
        str(round(summary["lowest_sales"])),
        summary["peak_date"],
        summary["lowest_date"]
    ]

    normalized_output = output.replace(",", "")

    for value in required_values:
        if value.replace(",", "") not in normalized_output:
            return False

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", output)
        if p.strip()
    ]

    if len(paragraphs) != 2:
        return False

    return True

def explain_forecast(summary):

    prompt = f"""
You are the AI explanation layer of a sales forecasting application.

Use ONLY these facts:

Forecast horizon: {summary['forecast_days']} days
Average forecasted sales: {summary['average_forecast']:.0f}
Highest forecasted sales: {summary['peak_sales']:.0f}
Highest forecast date: {summary['peak_date']}
Lowest forecasted sales: {summary['lowest_sales']:.0f}
Lowest forecast date: {summary['lowest_date']}

Write exactly two short paragraphs.

Paragraph 1:
Describe the forecast horizon, average forecast,
highest forecast and lowest forecast.

Paragraph 2:
Say only that variation between the highest and lowest
forecast periods may warrant additional business review.

Rules:
- Use only the supplied facts.
- Do not calculate new statistics.
- Do not calculate percentages.
- Do not infer demand.
- Do not mention inventory.
- Do not mention staffing.
- Do not mention revenue or profit.
- Do not say units sold.
- Do not invent causes.
- Do not invent recommendations.
- Do not add an introduction.
- Do not use headings.
- Do not output raw lists of numbers.
"""

    process = subprocess.run(
        ["ollama", "run", "llama3.2"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    if process.returncode != 0:
        return build_fallback_explanation(summary)

    output = process.stdout

    # Remove ANSI / terminal control sequences
    ansi_pattern = re.compile(
        r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
    )

    output = ansi_pattern.sub("", output)

    # Remove remaining non-printable characters
    output = "".join(
        char
        for char in output
        if char.isprintable() or char in "\n\t"
    )

    output = output.strip()

    # If Ollama violates our rules, use a safe factual fallback
    if not is_valid_explanation(output, summary):
        return build_fallback_explanation(summary)

    return output


if __name__ == "__main__":

    test_summary = {
        "forecast_days": 16,
        "average_forecast": 795115,
        "peak_sales": 1010263,
        "peak_date": "2017-08-27",
        "lowest_sales": 671104,
        "lowest_date": "2017-08-24"
    }

    print(explain_forecast(test_summary))