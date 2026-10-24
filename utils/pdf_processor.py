import pdfplumber


def extract_pdf_pages(pdf_file):
    """
    Extract text page-by-page.
    Returns a list containing page number and text.
    """

    pages = []

    with pdfplumber.open(pdf_file) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():

                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

    return pages


def create_chunks(pages, chunk_size=800, overlap=150):
    """
    Split document text into overlapping chunks.
    """

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text.strip(),
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks