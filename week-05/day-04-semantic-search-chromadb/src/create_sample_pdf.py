"""Create a small sample PDF for the W5D3 retrieval-augmented generation demo."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

out = Path(__file__).resolve().parent.parent / "sample_data" / "vector_databases_guide.pdf"
out.parent.mkdir(parents=True, exist_ok=True)

pages = [
    [
        "Vector Databases and Retrieval-Augmented Generation",
        "A vector embedding represents the semantic content of text as a numerical vector.",
        "An embedding model maps similar meanings to nearby points in vector space.",
        "Cosine similarity compares the direction of vectors. ChromaDB can use cosine distance",
        "to retrieve documents whose meanings are similar to a query.",
    ],
    [
        "Preparing Documents",
        "PDF text extraction converts document pages into text that can be searched.",
        "Chunking divides long text into smaller passages. Overlapping chunks can preserve",
        "context between adjacent passages. Each chunk can store metadata such as source filename",
        "and page number to support traceability and filtering.",
    ],
    [
        "Retrieval-Augmented Generation",
        "RAG first retrieves relevant document chunks for a user's question.",
        "The retrieved chunks are then included as context in a prompt to a language model.",
        "The model should answer using that context and acknowledge when the context does not",
        "contain enough information. Retrieval improves grounding but does not guarantee truth.",
    ],
]
pdf = canvas.Canvas(str(out), pagesize=letter)
width, height = letter
for lines in pages:
    y = height - 72
    for index, line in enumerate(lines):
        pdf.setFont("Helvetica-Bold" if index == 0 else "Helvetica", 14 if index == 0 else 11)
        pdf.drawString(60, y, line)
        y -= 32
    pdf.showPage()
pdf.save()
print(f"Created sample PDF: {out}")

