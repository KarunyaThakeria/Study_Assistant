
import fitz
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """Count tokens in a text string."""
    return len(encoding.encode(text))


# --------------------------------------------------
# EXTRACT TEXT FROM PDF
# --------------------------------------------------

def extract_pdf_text(pdf_file):
    """
    Extract text from every page of an uploaded PDF.

    Returns:
        List of dictionaries containing page_number and text.
    """

    pdf_bytes = pdf_file.getvalue()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            if text.strip():

                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    finally:
        document.close()

    return pages


# --------------------------------------------------
# SPLIT PDF INTO CHUNKS
# --------------------------------------------------

def split_into_chunks(
    pages,
    chunk_size=800,
    chunk_overlap=150
):
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk retains its original page number.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    all_chunks = []

    for page in pages:

        chunks = splitter.split_text(page["text"])

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "page_number": page["page_number"],
                "chunk_number": chunk_number,
                "token_count": count_tokens(chunk)
            })

    return all_chunks


# --------------------------------------------------
# PROCESS COMPLETE PDF
# --------------------------------------------------

def process_pdf(pdf_file):
    """
    Extract text and generate chunks for one PDF.

    Returns:
        pages, chunks
    """

    pages = extract_pdf_text(pdf_file)

    if not pages:

        raise ValueError(
            "No extractable text found. "
            "The PDF may be scanned or image-based. "
            "OCR support is required for scanned PDFs."
        )

    chunks = split_into_chunks(pages)

    if not chunks:

        raise ValueError(
            "Could not create text chunks from this PDF."
        )

    return pages, chunks
