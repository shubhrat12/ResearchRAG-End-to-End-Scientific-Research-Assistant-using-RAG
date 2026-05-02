import fitz  # PyMuPDF
import torch
import logging
import re
from typing import List, Dict
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512
FIGURE_LABEL_ID = 4  # Update this to match the label ID from your fine-tuned model

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
            encoding_raw = tokenizer(
                page_words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            token_to_word = encoding_raw.word_ids(batch_index=0)

            boxes = page_boxes[:MAX_LENGTH]
            while len(boxes) < MAX_LENGTH:
                boxes.append([0, 0, 0, 0])

            encoding_raw["bbox"] = torch.tensor([boxes], dtype=torch.long)
            encoding_raw["token_type_ids"] = torch.zeros_like(encoding_raw["input_ids"])
        except Exception as e:
            logger.error(f"❌ Tokenization failed on page {page.number}")
            logger.exception(e)
            all_page_detections.append([])
            continue

        encoding = {k: v.to(device) for k, v in encoding_raw.items()}
        with torch.no_grad():
            outputs = model(**encoding)

        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

        page_detections = []
        used_word_indices = set()

        for token_idx, pred in enumerate(predictions):
            word_idx = token_to_word[token_idx]
            if word_idx is None or word_idx >= len(page_words) or word_idx in used_word_indices:
                continue

            if pred == FIGURE_LABEL_ID and re.match(r'^(Fig\.?|Figure)$', page_words[word_idx], re.IGNORECASE):
                caption_words = [page_words[word_idx]]
                caption_boxes = [page_boxes[word_idx]]
                used_word_indices.add(word_idx)

                for i in range(word_idx + 1, min(len(page_words), word_idx + 15)):
                    if page_words[i] in [".", ",", ";", ":"]:
                        break
                    caption_words.append(page_words[i])
                    caption_boxes.append(page_boxes[i])
                    used_word_indices.add(i)

                caption_text = " ".join(caption_words)
                x0 = min(b[0] for b in caption_boxes)
                y0 = min(b[1] for b in caption_boxes)
                x1 = max(b[2] for b in caption_boxes)
                y1 = max(b[3] for b in caption_boxes)

                page_detections.append({
                    "caption": caption_text,
                    "bbox": [x0, y0, x1, y1]
                })

        logger.info(f"✅ Page {page.number}: Detected {len(page_detections)} figure captions")
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
            print(f"  - {f['caption']} @ {f['bbox']}")
