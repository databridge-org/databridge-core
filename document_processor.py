from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Union
from PIL.Image import Image
import json
import time
import subprocess

from dotenv import load_dotenv

load_dotenv(override=True)


# Base element classes
@dataclass
class DocumentElement:
    element_type: str
    content: Union[str, Image]
    page_number: int
    bounding_box: Optional[List[float]] = None  # [ymin, xmin, ymax, xmax]
    confidence: Optional[float] = None
    metadata: Optional[dict] = None


@dataclass
class ImageElement(DocumentElement):
    def __post_init__(self):
        self.element_type = "image"


@dataclass
class FormulaElement(DocumentElement):
    is_inline: bool = False

    def __post_init__(self):
        self.element_type = "formula"


@dataclass
class DiagramElement(DocumentElement):
    def __post_init__(self):
        self.element_type = "diagram"


@dataclass
class TableElement(DocumentElement):
    structure: Optional[dict] = None

    def __post_init__(self):
        self.element_type = "table"


@dataclass
class TextElement(DocumentElement):
    contains_formulas: bool = False

    def __post_init__(self):
        self.element_type = "text"


class DocumentProcessor:
    def __init__(self):
        self._system_prompt = """
        You are a professional document analysis system. Process the document and identify these elements:
        1. Images - Any graphical elements or photographs
        2. Formulas - Mathematical equations, both block and inline
        3. Diagrams - Flowcharts, graphs, technical drawings
        4. Tables - Tabular data with rows and columns
        5. Text - Paragraphs and headings, with inline formulas converted to LaTeX
        
        For text elements:
        - Preserve original text structure
        - Convert inline formulas to LaTeX wrapped in <latex> tags
        - Maintain reading order
        - Preserve special characters and formatting
        
        For all elements:
        - Provide bounding boxes in [ymin, xmin, ymax, xmax] format (0-1000 scale)
        - Include confidence scores
        - Categorize elements precisely
        """

        self._element_mapping = {
            "image": ImageElement,
            "formula": FormulaElement,
            "diagram": DiagramElement,
            "table": TableElement,
            "text": TextElement,
        }

    def process(self, document_path: Union[str, Path]) -> List[DocumentElement]:
        """Main processing method for documents"""
        document_path = Path(document_path)
        elements = []

        # Convert document to PDF if needed
        if document_path.suffix.lower() != ".pdf":
            pdf_path = self._convert_to_pdf(document_path)
        else:
            pdf_path = document_path

        # Convert PDF to images
        pages = self._convert_to_images(pdf_path)

        # Process each page
        for page_num, page_image in enumerate(pages, 1):
            page_elements = self._extract_elements(page_image, page_num)
            elements.extend(page_elements)

        return elements

    def _extract_elements(self, image: Image, page_num: int) -> List[DocumentElement]:
        """Extract elements from a single page image"""
        prompt = f"""Output JSON list of elements with fields:
        - "type": one of {list(self._element_mapping.keys())}
        - "content": text/Latex for formulas/text, null for others
        - "bounding_box": [ymin, xmin, ymax, xmax] (0-1000 scale)
        - "confidence": 0.0-1.0
        - "metadata": type-specific info
        
        For text:
        - "content" should contain text with inline formulas as <latex>equation</latex>
        - "metadata" should include "contains_formulas": boolean
        
        For formulas:
        - "is_inline": boolean
        - "content" should be LaTeX
        
        For tables:
        - "structure" should describe table format
        
        Be precise with bounding boxes and types."""

        response = self._call_gemini_api(image, prompt)
        return self._parse_response(response, page_num, image.size)

    def _call_gemini_api(self, image: Image, prompt: str, retries=3):
        """Helper method with retry logic"""
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[image, prompt],
                )
                return response.text
            except Exception:  # Using bare except to silence linter
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        return None

    def _parse_response(
        self, response: str, page_num: int, image_size: tuple
    ) -> List[DocumentElement]:
        """Parse Gemini response into DocumentElement objects"""
        try:
            parsed = json.loads(self._clean_json_response(response))
            elements = []

            for item in parsed:
                elem_type = item.get("type", "").lower()
                if elem_type not in self._element_mapping:
                    continue

                # Convert bounding box coordinates
                bbox = item.get("bounding_box", [])
                if len(bbox) != 4:
                    continue

                # Create appropriate element
                element_class = self._element_mapping[elem_type]
                element = element_class(
                    element_type=elem_type,  # Will be overridden in __post_init__
                    content=item.get("content", ""),
                    page_number=page_num,
                    bounding_box=bbox,
                    confidence=item.get("confidence", 1.0),
                    metadata=item.get("metadata", {}),
                )

                elements.append(element)

            return elements
        except json.JSONDecodeError:
            return []

    def _clean_json_response(self, response: str) -> str:
        """Extract JSON from markdown response"""
        if "```json" in response:
            return response.split("```json")[1].split("```")[0]
        return response

    def _convert_to_pdf(self, input_path: Path) -> Path:
        """Convert a document to PDF using LibreOffice in headless mode."""
        print(f"Converting {input_path} to PDF...")
        output_dir = Path("gem_output")
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
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )

            # The output PDF will have the same name as input but with .pdf extension
            output_pdf = output_dir / f"{input_path.stem}.pdf"

            if output_pdf.exists():
                print(f"Successfully converted {input_path} to PDF")
                return output_pdf
            else:
                raise FileNotFoundError(f"PDF was not created at expected path: {output_pdf}")

        except subprocess.CalledProcessError as e:
            print(f"Error converting document: {e.stderr}")
            raise
        except Exception as e:
            print(f"Unexpected error during conversion: {str(e)}")
            raise

    def _convert_to_images(self, pdf_path: Path) -> List[Image]:
        """Convert a PDF to a list of images."""
        print(f"Converting PDF {pdf_path} to images...")
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path))
        print(f"Successfully converted PDF to {len(images)} images")
        return images
