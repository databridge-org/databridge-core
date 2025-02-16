import subprocess
import time
from typing import Dict, List, Union
from pathlib import Path
from PIL.Image import Image
from PIL import ImageDraw, ImageFont
from PIL import ImageColor
import json


def convert_to_pdf(input_path: Union[str, Path], output_dir: str) -> str:
    """
    Convert a document to PDF using LibreOffice in headless mode.

    Args:
        input_path: Path to the input document
        output_dir: Directory where the PDF should be saved

    Returns:
        str: Path to the generated PDF file
    """
    print(f"Converting {input_path} to PDF...")
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
        print("Running LibreOffice conversion...")
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


from pdf2image import convert_from_path


def convert_to_images(pdf_path: str) -> List[Image]:
    """Convert a PDF to a list of images."""
    print(f"Converting PDF {pdf_path} to images...")
    images = convert_from_path(pdf_path)
    print(f"Successfully converted PDF to {len(images)} images")
    return images


from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# bounding_box_system_instructions = """
#     Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
#     If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
#       """


def extract_image_bounding_boxes(image: Image, items: str, label_instructions: str) -> Dict:
    """Extract images from PDF and save them with contextual information."""
    print(f"Extracting bounding boxes for {items}...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"""Detect {items}. Output a json list where each entry contains the 2D bounding box in "box_2d" and the {label_instructions} in "label". The bounding box should be in the format [ymin, xmin, ymax, xmax] where values are between 0 and 1000. If you don't see any of the items, return an empty json output - but always return a json output."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",  # "gemini-2.0-flash", # "gemini-2.0-pro-exp",
        contents=[image, prompt],
    )
    print("Successfully extracted bounding boxes")
    return response.text


def parse_json(json_output):
    """Parse JSON output from markdown format."""
    print("Parsing JSON output...")
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    print("Successfully parsed JSON")
    return json_output


additional_colors = [colorname for (colorname, _) in ImageColor.colormap.items()]


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """
    print("Plotting bounding boxes on image...")
    # Load the image
    img = im
    width, height = img.size
    print(f"Image dimensions: {width}x{height}")
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    # Iterate over the bounding boxes
    boxes_drawn = 0
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
        # Select a color from the list
        color = colors[i % len(colors)]

        if len(bounding_box["box_2d"]) != 4:
            print(f"Bounding box {i} has {len(bounding_box['box_2d'])} coordinates - skipping")
            continue

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color)  # , font=font)
        boxes_drawn += 1

    print(f"Drew {boxes_drawn} bounding boxes")
    # Display the image
    img.show()

    return img


def extract_diagrams_from_bounding_boxes(img: Image, bounding_boxes: str):
    print("Extracting diagrams from bounding boxes...")
    width, height = img.size
    bounding_boxes = json.loads(bounding_boxes)
    cropped_imgs = []

    for i, bounding_box in enumerate(bounding_boxes):
        if len(bounding_box["box_2d"]) != 4:
            print(f"Bounding box {i} has {len(bounding_box['box_2d'])} coordinates - skipping")
            continue
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = max(0, int(bounding_box["box_2d"][0] / 1000 * height) - 30)
        abs_x1 = max(0, int(bounding_box["box_2d"][1] / 1000 * width) - 30)
        abs_y2 = min(height, int(bounding_box["box_2d"][2] / 1000 * height) + 30)
        abs_x2 = min(width, int(bounding_box["box_2d"][3] / 1000 * width) + 30)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Extract (crop) the region defined by the bounding box
        cropped_img = img.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        cropped_imgs.append(cropped_img)
        print(f"Extracted diagram {i+1}")

    print(f"Successfully extracted {len(cropped_imgs)} diagrams")
    return cropped_imgs


if __name__ == "__main__":
    document_name = "/Users/adi/Downloads/documents/62250266_origin.pdf"
    # document_name = "samples/Documents/a.pdf"
    print(f"Processing document: {document_name}")
    images = convert_to_images(document_name)
    for page_num, test_image in enumerate(images):
        print(f"\nProcessing page {page_num + 1}")
        while True:
            try:
                bounding_boxes = extract_image_bounding_boxes(
                    test_image,
                    "diagrams, images, mathematical formulas, and other non-text elements",
                    "label each element with the term 'diagram', 'image', 'formula', or 'other'",
                )
                break
            except Exception as e:
                print(f"Error extracting bounding boxes: {e}")
                print("Retrying extraction after 4 seconds...")
                time.sleep(4)
                continue

        bounding_boxes = parse_json(bounding_boxes)
        # print(bounding_boxes)
        # Save bounding boxes to output file
        plot_bounding_boxes(test_image, bounding_boxes)
        print(bounding_boxes)
        diagrams = extract_diagrams_from_bounding_boxes(test_image, bounding_boxes)
        # Create output directory based on document name
        doc_dir = Path(document_name).stem
        output_dir = Path(doc_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {output_dir}")

        # Save each diagram with page and diagram number
        for diagram_num, diagram in enumerate(diagrams, 1):
            output_path = output_dir / f"page_{page_num+1}_diagram_{diagram_num}.png"
            diagram.save(output_path)
            print(f"Saved diagram to {output_path}")

        # output_file = "bounding_boxes_output.json"
        # with open(output_file, "w") as f:
        #     json.dump(bounding_boxes, f, indent=2)
        # print(f"Saved bounding boxes to {output_file}")

        output_file = output_dir / f"page_{page_num+1}_bounding_boxes.json"
        with open(output_file, "w") as f:
            json.dump(bounding_boxes, f, indent=2)
        print(f"Saved bounding boxes to {output_file}")
