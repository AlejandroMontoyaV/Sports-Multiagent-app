from src.utils.pdf_to_text import pdf_to_text

file_name = "hola.txt"
file_path = "data/new_docs/hola.txt"
docs_path = "data/raw_docs"
txt_path = pdf_to_text(file_name, file_path, docs_path)
print(f"Archivo final en: {txt_path}")