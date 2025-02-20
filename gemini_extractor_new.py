import os
from pathlib import Path
from typing import List
from PIL.Image import Image
from PIL import ImageDraw, ImageColor
import json

from document_processor import DocumentProcessor, DocumentElement, DiagramElement


class GeminiExtractor:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.additional_colors = [colorname for (colorname, _) in ImageColor.colormap.items()]

    def process_document(self, document_path: Path) -> List[DocumentElement]:
        """Process document and extract elements"""
        return self.processor.process(document_path)

    def plot_element_boxes(self, image: Image, elements: List[DocumentElement]) -> Image:
        """Visualize elements on the page image"""
        draw = ImageDraw.Draw(image)
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
        ] + self.additional_colors

        for idx, element in enumerate(elements):
            if not element.bounding_box or len(element.bounding_box) != 4:
                continue

            color = colors[idx % len(colors)]
            width, height = image.size

            # Convert normalized coordinates to absolute
            ymin, xmin, ymax, xmax = element.bounding_box
            abs_coords = (
                int(xmin * width / 1000),
                int(ymin * height / 1000),
                int(xmax * width / 1000),
                int(ymax * height / 1000),
            )

            # Draw rectangle and label
            draw.rectangle(abs_coords, outline=color, width=4)

            # Create label with content preview
            label = f"{element.element_type} {idx+1}"
            if element.content:
                preview = (
                    str(element.content)[:50] + "..."
                    if len(str(element.content)) > 50
                    else str(element.content)
                )
                label = f"{label}: {preview}"

            # Draw label with background for better visibility
            text_bbox = draw.textbbox((abs_coords[0] + 8, abs_coords[1] + 6), label)
            draw.rectangle(text_bbox, fill="white", outline=color)
            draw.text((abs_coords[0] + 8, abs_coords[1] + 6), label, fill=color)

        return image

    def extract_diagrams(self, image: Image, elements: List[DocumentElement]) -> List[Image]:
        """Extract diagram elements as cropped images"""
        diagrams = []
        width, height = image.size

        for element in elements:
            if not isinstance(element, DiagramElement) or not element.bounding_box:
                continue

            ymin, xmin, ymax, xmax = element.bounding_box
            abs_coords = (
                max(0, int(xmin * width / 1000) - 30),
                max(0, int(ymin * height / 1000) - 30),
                min(width, int(xmax * width / 1000) + 30),
                min(height, int(ymax * height / 1000) + 30),
            )

            if abs_coords[0] >= abs_coords[2] or abs_coords[1] >= abs_coords[3]:
                continue  # Skip invalid boxes

            diagrams.append(image.crop(abs_coords))

        return diagrams


if __name__ == "__main__":
    document_path = "/Users/adi/Downloads/documents/62250266_origin.pdf"
    extractor = GeminiExtractor()

    try:
        # Create output directory
        output_dir = Path("gem_output")
        output_dir.mkdir(exist_ok=True)

        # Process document and get elements
        elements = extractor.process_document(document_path)

        # Convert PDF to images using processor
        images = extractor.processor._convert_to_images(document_path)

        for page_num, page_image in enumerate(images, 1):
            print(f"\nProcessing page {page_num}")
            page_elements = [e for e in elements if e.page_number == page_num]

            # Generate and save visualization
            annotated_img = extractor.plot_element_boxes(page_image.copy(), page_elements)
            vis_path = output_dir / f"page_{page_num}_annotated.png"
            annotated_img.save(vis_path)
            print(f"Saved visualization to {vis_path}")

            # Extract and save diagrams
            diagrams = extractor.extract_diagrams(page_image, page_elements)
            for idx, diagram in enumerate(diagrams, 1):
                diagram_path = output_dir / f"page_{page_num}_diagram_{idx}.png"
                diagram.save(diagram_path)
                print(f"Saved diagram to {diagram_path}")

            # Save bounding boxes JSON
            bbox_data = [
                {
                    "type": e.element_type,
                    "label": getattr(e, "label", ""),
                    "bounding_box": e.bounding_box,
                    "confidence": e.confidence,
                }
                for e in page_elements
            ]

            json_path = output_dir / f"page_{page_num}_bounding_boxes.json"
            with open(json_path, "w") as f:
                json.dump(bbox_data, f, indent=2)
            print(f"Saved bounding boxes to {json_path}")

    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise
