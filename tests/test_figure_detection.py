import fitz  # PyMuPDF
import torch
import logging
from typing import List, Dict
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512
FIGURE_LABEL_ID = 4

def detect_figures_in_pdf(pdf_path: str, tokenizer, model) -> List[List[Dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    doc = fitz.open(pdf_path)
    all_page_detections = []

    for page in doc:
        words = page.get_text("words")
        if not words:
            all_page_detections.append([])
            continue

        page_words = []
        page_boxes = []

        for w in words:
            x0, y0, x1, y1, text, *_ = w
            if text.strip():
                page_words.append(text)
                page_boxes.append([int(x0), int(y0), int(x1), int(y1)])

        if not page_words:
            all_page_detections.append([])
            continue

        try:
            encoding = tokenizer(
                page_words,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )

            # Manually add bounding boxes if needed
            # Ensure the number of boxes matches max_length
            boxes = page_boxes[:MAX_LENGTH]
            while len(boxes) < MAX_LENGTH:
                boxes.append([0, 0, 0, 0])
            encoding['bbox'] = torch.tensor([boxes], dtype=torch.long)

        except Exception as e:
            logger.error(f"❌ Tokenization failed on page {page.number}")
            logger.exception(e)
            all_page_detections.append([])
            continue

        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)

        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        word_ids = encoding['input_ids'].squeeze().tolist()

        page_detections = []
        for token_idx, pred in enumerate(predictions):
            if pred == FIGURE_LABEL_ID and token_idx < len(page_words):
                page_detections.append({
                    "text": page_words[token_idx],
                    "bbox": page_boxes[token_idx]
                })

        logger.info(f"✅ Page {page.number}: Detected {len(page_detections)} figure tokens")
        all_page_detections.append(page_detections)

    doc.close()
    return all_page_detections

# === RUN TEST ===
if __name__ == "__main__":
    model_path = "models/figure_detector/best_model"

    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
    model = LayoutLMForTokenClassification.from_pretrained(model_path)

    pdf_path = "D:/Langchain Project/downloads/1.pdf"
    detections = detect_figures_in_pdf(pdf_path, tokenizer, model)

    for page_num, figures in enumerate(detections):
        print(f"\nPage {page_num}:")
        if not figures:
            print("  No figures detected.")
        for f in figures:
            print(f"  - {f['text']} @ {f['bbox']}")
