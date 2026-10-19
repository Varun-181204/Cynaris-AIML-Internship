
"""Generate W4D6 documentation, notebook, and model-card PDF."""

from pathlib import Path
from xml.sax.saxutils import escape

import nbformat as nbf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NOTEBOOKS = ROOT / "notebooks"
REPORTS = ROOT / "reports"

for directory in (DOCS, NOTEBOOKS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)


MODEL_CARD = """# Model Card: W4D1 Binary Classification Model

## 1. Model details
- Model: StandardScaler followed by Logistic Regression
- Task: Binary classification
- Framework: scikit-learn Pipeline
- Random seed: 42
- Original exercise: Cynaris AI/ML Internship, Week 4 Day 1

## 2. Intended use
This model demonstrates reproducible model training, preprocessing, evaluation, cross-validation, and explanation. It is intended for educational experimentation only. It must not be used to make automated high-impact decisions.

## 3. Training data
The model uses a synthetic dataset generated with scikit-learn's make_classification: 1,500 samples, 20 numerical features, 8 informative features, 4 redundant features, and an approximate 70/30 class balance. The dataset contains no real individuals' records.

## 4. Training procedure
The data is split into training and test sets using stratification and random seed 42. StandardScaler and LogisticRegression are combined in a Pipeline. Scaling is fitted on training data only. The original exercise uses max_iter=2000.

## 5. Evaluation results
- Test accuracy: 0.8067
- Test precision: 0.8511
- Test recall: 0.4396
- Test F1-score: 0.5797
- Test ROC-AUC: 0.8393
- Stratified five-fold cross-validation ROC-AUC: 0.8114 ± 0.0289

These values are from the W4D1 evaluation report. The SHAP demonstration refits the same model configuration on a stratified split; its split accuracy is recorded separately in shap_summary_metrics.json.

## 6. Fairness and explainability
The W4D6 exercise separately evaluates gender-related group metrics on the UCI Adult Income dataset and demonstrates AIF360 Reweighing. SHAP is used to explain the W4D1 synthetic model. Adult Income results are not a fairness audit of the W4D1 model.

## 7. Limitations and ethical considerations
Synthetic features do not represent real-world demographic characteristics. Consequently, demographic fairness cannot be established for the W4D1 model. The model has not been validated for hiring, lending, healthcare, policing, or other high-impact use. SHAP describes model behavior but does not prove causation or eliminate bias.

## 8. Monitoring and future work
Future work should use a documented, representative dataset, assess subgroup performance where appropriate protected attributes are available, conduct independent reviews, monitor performance over time, retain human oversight, and provide a way to challenge or correct harmful decisions.
"""


INDIA_CASE = """# India AI-Bias Case Study: Facial Recognition

## Case and risk

NITI Aayog's November 2022 discussion paper, *Responsible AI for All: Adopting the Framework — A Use Case Approach on Facial Recognition Technology*, discusses risks from facial-recognition technology (FRT) in India. The paper reports a study of four commercial FRT systems tested against Indian electoral rolls, with an average identification error rate reported as 0.5% for Indian men and 3% for Indian women. This reported gap illustrates how overall accuracy can conceal unequal error rates across demographic groups.

## Possible causes

Training and evaluation datasets may not adequately represent the diversity of India's population. Image quality, lighting, camera position, and system thresholds can also affect performance. The reported difference is a reason for further investigation; it does not, by itself, prove one specific cause or describe every FRT system.

## Recommended safeguards

Before deployment, vendors and public agencies should independently test representative Indian datasets and publish subgroup false-match and false-non-match rates. Testing should cover varied lighting, ages, and relevant demographic groups. Audits should be repeated after major updates. In high-impact situations, a match should not be treated as conclusive proof of identity: trained human review and an appeal process are necessary. Organisations should also explain the purpose of collection, limit retention and access, protect biometric data, and assign responsibility for errors.

## Conclusion

The case demonstrates why responsible AI needs subgroup evaluation, transparency, privacy protection, accountability, and meaningful human oversight—not merely a strong aggregate accuracy score.

## Reference

NITI Aayog. (November 2022). *Responsible AI for All: Adopting the Framework — A Use Case Approach on Facial Recognition Technology*. https://www.niti.gov.in/sites/default/files/2022-11/Ai_for_All_2022_02112022.pdf
"""


def create_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# W4D6: Responsible AI — Bias, Fairness and Transparency\n\n"
            "This notebook evaluates gender-related group fairness on the "
            "UCI Adult Income dataset using AIF360, compares a baseline "
            "classifier with Reweighing, and displays SHAP explanations "
            "for the separate W4D1 synthetic model.\n\n"
            "**Important:** these are two separate experiments. The Adult "
            "Income fairness results do not describe the W4D1 model."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Fairness metrics\n\n"
            "- **Disparate impact:** ratio of positive prediction rates "
            "for the unprivileged and privileged groups.\n"
            "- **Demographic parity difference:** difference between "
            "group positive prediction rates.\n"
            "- **Average odds difference:** combines differences in "
            "false-positive and true-positive rates.\n"
            "- **Equal opportunity difference:** difference in true-positive "
            "rates.\n\n"
            "The 0.80 disparate-impact value is a screening heuristic, "
            "not a universal legal rule or proof of fairness."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import importlib.util\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "candidates = [Path.cwd(), *Path.cwd().parents]\n"
            "project_root = next(\n"
            "    p for p in candidates\n"
            "    if (p / 'week-04/day-06-responsible-ai/src/"
            "responsible_ai_audit.py').exists()\n"
            ")\n"
            "audit_path = project_root / 'week-04/day-06-responsible-ai/"
            "src/responsible_ai_audit.py'\n"
            "spec = importlib.util.spec_from_file_location('responsible_ai_audit', audit_path)\n"
            "audit = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(audit)\n"
            "print('Project root:', project_root)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Baseline versus Reweighing\n\n"
            "Reweighing changes training sample weights to reduce "
            "group/label imbalance. The test set remains unchanged."
        ),
        nbf.v4.new_code_cell(
            "fairness_results = audit.audit_adult_income()\n"
            "display(fairness_results.round(4))"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(audit.OUTPUTS / 'fairness_comparison.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Interpretation\n\n"
            "Compare the baseline and reweighed results rather than "
            "looking at one metric alone. Reweighing may improve one "
            "fairness measure while leaving other disparities. Accuracy "
            "and fairness must both be considered, and results depend "
            "on dataset quality, the chosen groups, and the model."
        ),
        nbf.v4.new_markdown_cell(
            "## 3. SHAP explanation for the W4D1 model\n\n"
            "This separate experiment uses the synthetic classification "
            "dataset from W4D1. SHAP values describe how features contribute "
            "to this fitted model's predictions; they do not establish "
            "causation or real-world fairness."
        ),
        nbf.v4.new_code_cell(
            "shap_results = audit.explain_w4d1_model()\n"
            "display(pd.Series(shap_results, name='value').to_frame())\n"
            "display(Image(filename=str(audit.OUTPUTS / 'shap_summary.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Limitations and responsible use\n\n"
            "- The Adult Income dataset has missing values that AIF360 may remove.\n"
            "- Results are specific to this split, preprocessing, model, and group definitions.\n"
            "- Reweighing does not guarantee fairness across every metric or intersectional group.\n"
            "- SHAP explanations do not prove that a feature causes an outcome.\n"
            "- Neither experiment justifies deploying a model in a high-impact setting without further validation."
        ),
    ]

    path = NOTEBOOKS / "responsible_ai_audit.ipynb"
    nbf.write(notebook, path)
    print(f"Created notebook: {path}")


def create_model_card_pdf():
    """Render the Markdown model card as a readable PDF."""
    source = DOCS / "model_card.md"
    destination = REPORTS / "model_card.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CardTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#23395B"),
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="CardHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#23395B"),
        spaceBefore=10,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="CardBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=6,
    ))

    story = []
    bullet_items = []

    def flush_bullets():
        nonlocal bullet_items
        if bullet_items:
            story.append(ListFlowable(
                [
                    ListItem(
                        Paragraph(escape(item), styles["CardBody"]),
                        leftIndent=12,
                    )
                    for item in bullet_items
                ],
                bulletType="bullet",
                leftIndent=18,
                bulletFontSize=8,
            ))
            story.append(Spacer(1, 5))
            bullet_items = []

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:
            flush_bullets()
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(Paragraph(escape(line[2:]), styles["CardTitle"]))
        elif line.startswith("## "):
            flush_bullets()
            story.append(Paragraph(escape(line[3:]), styles["CardHeading"]))
        elif line.startswith("- "):
            bullet_items.append(line[2:])
        else:
            flush_bullets()
            story.append(Paragraph(escape(line), styles["CardBody"]))

    flush_bullets()

    pdf = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="W4D1 Model Card",
        author="Cynaris AI/ML Internship",
    )
    pdf.build(story)
    print(f"Created PDF: {destination}")


def main():
    (DOCS / "model_card.md").write_text(
        MODEL_CARD.strip() + "\n",
        encoding="utf-8",
    )
    (DOCS / "india_bias_case.md").write_text(
        INDIA_CASE.strip() + "\n",
        encoding="utf-8",
    )

    create_notebook()
    create_model_card_pdf()

    print("W4D6 documentation deliverables created.")


if __name__ == "__main__":
    main()
