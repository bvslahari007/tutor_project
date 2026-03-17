import pdfplumber
import os

def pdf_to_text(pdf_file, save_to):
    if not os.path.exists(pdf_file):
        print("Cannot find the PDF file. Check the name and location.")
        return

    print("Reading the PDF, please wait...")
    pages = []

    with pdfplumber.open(pdf_file) as book:
        total = len(book.pages)
        for num, page in enumerate(book.pages):
            content = page.extract_text()
            if content and len(content.strip()) > 80:
                pages.append(content.strip())
            if (num + 1) % 20 == 0:
                print(f"Done {num + 1} out of {total} pages")

    full = "\n\n".join(pages)

    with open(save_to, "w", encoding="utf-8") as f:
        f.write(full)

    print("Finished.")
    print(f"Pages saved: {len(pages)}")
    print(f"Saved to: {save_to}")

pdf_to_text("science_class10.pdf", "data/science_class10.txt")