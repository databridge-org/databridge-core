import argparse
import os
import subprocess
from typing import List, Dict, Union
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image


def convert_to_pdf(input_path: Union[str, Path], output_dir: str) -> str:
    """
    Convert a document to PDF using LibreOffice in headless mode.

    Args:
        input_path: Path to the input document
        output_dir: Directory where the PDF should be saved

    Returns:
        str: Path to the generated PDF file
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the LibreOffice command
    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        str(input_path),
        "--outdir",
        str(output_dir),
    ]

    try:
        # Run the conversion
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # The output PDF will have the same name as input but with .pdf extension
        output_pdf = output_dir / f"{input_path.stem}.pdf"

        if output_pdf.exists():
            print(f"Successfully converted {input_path} to PDF")
            return str(output_pdf)
        else:
            raise FileNotFoundError(f"PDF was not created at expected path: {output_pdf}")

    except subprocess.CalledProcessError as e:
        print(f"Error converting document: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error during conversion: {str(e)}")
        raise


def extract_images_with_context(pdf_path: str, output_dir: str) -> Dict:
    """Extract images from PDF and save them with contextual information."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing PDF: {pdf_path}")

    # Partition PDF with image extraction enabled
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",  # Required for image extraction
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=output_dir,
    )

    # Convert to list for easier processing
    elements_list = list(elements)

    # Extract all narrative text for full document content
    full_text = "\n".join(
        str(elem)
        for elem in elements_list
        if hasattr(elem, "category") and elem.category == "NarrativeText"
    )

    # Track images
    image_info = []

    # Process each element
    for element in elements_list:
        if isinstance(element, Image):
            image_info.append(
                {
                    "image_path": element.metadata.image_path,
                    "page_number": element.metadata.page_number,
                }
            )
            print(f"Found image on page {element.metadata.page_number}")

    # Save full document text
    text_file = Path(output_dir) / "full_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {"images": image_info, "full_text": full_text}


def main():
    parser = argparse.ArgumentParser(description="Extract images from PDF with context")
    parser.add_argument("input_paths", nargs="+", help="Path(s) to document file(s)")
    parser.add_argument(
        "--output-dir",
        default="extracted_images",
        help="Directory to save extracted images (default: extracted_images)",
    )
    parser.add_argument(
        "--convert", action="store_true", help="Convert non-PDF documents to PDF before processing"
    )

    args = parser.parse_args()

    total_images = 0
    for input_path in args.input_paths:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            continue

        try:
            pdf_path = input_path
            # If the file is not a PDF and conversion is requested, convert it
            if args.convert and not input_path.lower().endswith(".pdf"):
                pdf_path = convert_to_pdf(input_path, args.output_dir)

            if pdf_path.lower().endswith(".pdf"):
                image_info = extract_images_with_context(pdf_path, args.output_dir)
                num_images = len(image_info["images"])
                total_images += num_images
                print(f"\nExtracted {num_images} images from {pdf_path}")
            else:
                print(f"Skipping non-PDF file: {input_path}")

        except Exception as e:
            print(f"Error processing file {input_path}: {e}")

    print(f"\nTotal images extracted across all documents: {total_images}")


if __name__ == "__main__":
    main()
