import os
import fitz  # PyMuPDF

def convert_pdfs_to_images_pymupdf(pdf_folder, output_folder, zoom_x=2.0, zoom_y=2.0, rotation=0):
    """
    Convert all PDFs in a folder to images using PyMuPDF, saving pages as PNG files.

    Args:
        pdf_folder (str): Path to folder containing PDF files.
        output_folder (str): Path to folder to save output images.
        zoom_x (float): Horizontal zoom factor (default 2.0 for higher resolution).
        zoom_y (float): Vertical zoom factor.
        rotation (int): Rotation angle (default 0).
    """
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Processing {pdf_file}...")

        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_output_dir = os.path.join(output_folder, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # zero-based
            pix = page.get_pixmap(matrix=mat, alpha=False)  # Render page to image
            image_path = os.path.join(pdf_output_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)

        print(f"Saved {len(doc)} pages for {pdf_file} to {pdf_output_dir}")

if __name__ == "__main__":
    pdf_folder_path = "path/to/your/pdf/folder"
    images_output_folder = "path/to/save/images"

    convert_pdfs_to_images_pymupdf(pdf_folder_path, images_output_folder)
