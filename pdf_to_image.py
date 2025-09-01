import os
import fitz  # PyMuPDF

def convert_pdfs_to_images_pymupdf(pdf_folder, output_folder, zoom_x=2.0, zoom_y=2.0, rotation=0):
    import fitz  # PyMuPDF
    import os

    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Processing {pdf_file}...")
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(pdf_file)[0].strip()  # <-- Fix: Remove spaces
        pdf_output_dir = os.path.join(output_folder, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image_path = os.path.join(pdf_output_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)
        print(f"Saved {len(doc)} pages for {pdf_file} to {pdf_output_dir}")

if __name__ == "__main__":
    pdf_folder_path = "Trash - Eshan"
    images_output_folder = "data/train/bad"

    convert_pdfs_to_images_pymupdf(pdf_folder_path, images_output_folder)
