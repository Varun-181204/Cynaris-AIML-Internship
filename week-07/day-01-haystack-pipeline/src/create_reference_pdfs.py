from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "pdfs"
OUTPUT.mkdir(parents=True, exist_ok=True)

documents = {
    "haystack_pipeline_architecture.pdf": {
        "title": "Haystack Pipeline Architecture",
        "sections": [
            ("1. Introduction",
             "Haystack is an open-source framework for building "
             "modular AI applications, search systems and retrieval "
             "augmented generation (RAG) pipelines."),
            ("2. Pipeline Architecture",
             "A Haystack pipeline connects independent components "
             "through named inputs and outputs. A component can "
             "convert documents, preprocess text, write to a document "
             "store, retrieve relevant content or generate answers."),
            ("3. Document Conversion",
             "PDF converters extract textual content from PDF files "
             "and represent the result as Documents. Metadata such as "
             "filename and page number helps preserve source information."),
            ("4. Document Stores",
             "A document store manages documents and provides an "
             "interface for writing and retrieving them. InMemoryDocumentStore "
             "is useful for demonstrations and experiments. Persistent "
             "stores are preferable for production workloads."),
            ("5. Retrieval-Augmented Generation",
             "A typical RAG workflow converts source documents, indexes "
             "them, retrieves relevant passages for a user query and "
             "passes the retrieved context to a language model."),
            ("6. Pipeline Connections",
             "Pipeline connections must match component input and output "
             "types. Components should be tested independently before "
             "connecting the complete workflow."),
            ("7. Evaluation",
             "A retrieval system should be evaluated using relevant "
             "documents, ranking quality, retrieval latency and "
             "repeatable test queries. Retrieval quality and generated "
             "answer quality are separate evaluation concerns."),
            ("8. Best Practices",
             "Keep source metadata, avoid indexing duplicate documents, "
             "use consistent preprocessing, record model versions and "
             "save reproducible evaluation evidence."),
        ],
    },
    "retrieval_methods_and_evaluation.pdf": {
        "title": "Information Retrieval: BM25 and Dense Retrieval",
        "sections": [
            ("1. Information Retrieval",
             "Information retrieval identifies and ranks documents "
             "that are useful for a user's information need. Search "
             "systems commonly return the top K matching documents."),
            ("2. BM25 Retrieval",
             "BM25 is a sparse lexical ranking algorithm based on "
             "term frequency, inverse document frequency and document "
             "length normalization. It is effective when query terms "
             "overlap with relevant document terms."),
            ("3. BM25 Limitations",
             "BM25 may fail when a query uses synonyms or different "
             "wording from the source document. It primarily measures "
             "lexical matching rather than semantic similarity."),
            ("4. Dense Retrieval",
             "Dense retrieval converts documents and queries into "
             "numeric embedding vectors. A similarity function such "
             "as cosine similarity or dot product ranks documents "
             "according to their vector representations."),
            ("5. Embeddings",
             "An embedding model maps text into a vector space. "
             "Texts with related meanings may have nearby vectors, "
             "even when they do not share the same exact words. "
             "Query and document embeddings must use compatible models."),
            ("6. Precision at K",
             "Precision at K is the number of relevant retrieved "
             "documents among the first K results divided by K. "
             "For example, if 3 of the top 5 results are relevant, "
             "Precision@5 equals 0.60."),
            ("7. Recall at K",
             "Recall at K is the number of relevant retrieved documents "
             "among the first K results divided by the total number "
             "of known relevant documents for that query."),
            ("8. Mean Reciprocal Rank",
             "Mean Reciprocal Rank averages the reciprocal rank of "
             "the first relevant result across queries. It rewards "
             "systems that place a relevant result near the top."),
            ("9. Fair Comparison",
             "BM25 and dense retrieval should use the same documents, "
             "the same ten queries, the same top K and the same "
             "human relevance judgments. Report per-query results "
             "as well as aggregate metrics."),
            ("10. Limitations",
             "Human relevance judgments can be subjective. A small "
             "dataset may not represent real-world search behavior. "
             "Embedding quality depends on the selected model and "
             "the domain of the documents."),
        ],
    },
}

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocumentTitle",
    parent=styles["Title"],
    fontSize=19,
    leading=25,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#243B53"),
    spaceAfter=22,
))
styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading2"],
    fontSize=12,
    leading=16,
    textColor=colors.HexColor("#245B78"),
    spaceBefore=12,
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="BodyTextCustom",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    spaceAfter=8,
))

for filename, content in documents.items():
    path = OUTPUT / filename

    story = [
        Paragraph(escape(content["title"]), styles["DocumentTitle"]),
        Paragraph(
            "Cynaris AI/ML Internship - Week 7 Reference Material",
            styles["Normal"],
        ),
        Spacer(1, 16),
    ]

    for heading, body in content["sections"]:
        story.append(
            Paragraph(escape(heading), styles["SectionHeading"])
        )
        story.append(
            Paragraph(escape(body), styles["BodyTextCustom"])
        )

    pdf = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=content["title"],
        author="Cynaris AI/ML Internship",
    )
    pdf.build(story)
    print(f"Created: {path.name}")

print("Reference PDF generation completed.")
