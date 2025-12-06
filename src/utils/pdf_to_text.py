# src/utils/file_ingestion.py  (por ejemplo)

import os
import shutil
import fitz  # PyMuPDF -> pip install pymupdf


def pdf_to_text(file_name: str, file_path: str, raw_docs_path: str) -> str:
    """
    Procesa un archivo subido por el usuario.

    - Si es PDF: lo convierte a .txt y guarda el .txt en raw_docs_path.
    - Si es TXT: simplemente lo mueve a raw_docs_path.
    - En ambos casos elimina el archivo original de la carpeta temporal/new_docs.

    Parámetros
    ----------
    file_name : str
        Nombre original del archivo (ej: "documento.pdf").
    file_path : str
        Ruta completa donde se guardó temporalmente (ej: "data/new_docs/documento.pdf").
    raw_docs_path : str
        Carpeta destino donde deben quedar los .txt definitivos.

    Devuelve
    --------
    str
        Ruta completa del archivo .txt final en raw_docs_path.
    """

    os.makedirs(raw_docs_path, exist_ok=True)

    base_name, ext = os.path.splitext(file_name)
    ext = ext.lower()

    # Caso 1: PDF -> convertir a TXT
    if ext == ".pdf":
        txt_name = base_name + ".txt"
        txt_dest_path = os.path.join(raw_docs_path, txt_name)

        # Extraer texto con PyMuPDF
        text = ""
        with fitz.open(file_path) as pdf_doc:
            for page in pdf_doc:
                text += page.get_text()

        # Guardar el texto en el destino final
        with open(txt_dest_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Eliminar el PDF original de new_docs
        os.remove(file_path)

        return txt_dest_path

    # Caso 2: TXT -> mover directamente
    elif ext == ".txt":
        txt_dest_path = os.path.join(raw_docs_path, file_name)
        shutil.move(file_path, txt_dest_path)
        return txt_dest_path

    # Otros formatos:
    else:
        raise ValueError(f"Formato de archivo no soportado: {ext}")
