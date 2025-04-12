# 📂 Imports
import os
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
import pytesseract

# 📄 Get PDF from pdfs directory
pdfs_dir = "pdfs"
pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]

if not pdf_files:
    print("No PDF files found in the 'pdfs' directory. Please add your PDF files there.")
    exit(1)

# Use the first PDF found (you can modify this to handle multiple files)
pdf_path = os.path.join(pdfs_dir, pdf_files[0])
print(f"Processing PDF: {pdf_path}")

# 🗾️ Convert PDF to images
os.makedirs("pages", exist_ok=True)
pages = convert_from_path(pdf_path, dpi=200)
image_paths = []
for i, page in enumerate(pages):
    path = f"pages/page_{i+1}.png"
    page.save(path, "PNG")
    image_paths.append(path)

print(f"Converted {len(image_paths)} pages to images.")

# 🧠 Load Donut model (OCR-free DocVQA)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# 📝 Set prompt
prompt = "<s_docvqa><question>List all construction materials mentioned in the page.</question><answer>"

# 🔍 Run inference on first page (you can loop this)
image = Image.open(image_paths[0]).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
decoder_input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids

output = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

# 📋 Show result
display(image)
print("📦 Materials extracted:")
print(decoded)