import fitz  # PyMuPDF
import os

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def save_uploaded_files(uploaded_files, save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    paths = []
    for file in uploaded_files:
        path = os.path.join(save_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(path)
    return paths
