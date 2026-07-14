from pathlib import Path

from docling.document_converter import DocumentConverter

converter = DocumentConverter()

test_pdf = "/mnt/e/Sanjana_Code/ShopSphere/data/pdfs/ShopSphere_Employee_Handbook_v1.0.pdf"

result = converter.convert(test_pdf)

markdown = result.document.export_to_markdown()

print(markdown[:1000])