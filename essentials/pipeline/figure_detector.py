import fitz  # PyMuPDF
import torch
import logging
import re
from typing import List, Dict
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from pathlib import Path

# Set up logging to pipeline.log in the root directory only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(Path(__file__).resolve().parent.parent.parent / "pipeline.log"))
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 512
FIGURE_LABEL_ID = 4  # Assuming label 4 represents 'figure caption'

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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

def convert_figures_to_chunks(detections: List[List[Dict]], pdf_path: str) -> List[Dict]:
    """
    Convert the output of detect_figures_in_pdf to context-builder-compatible chunks.
    Each chunk will have 'text' (caption), 'metadata' (content_type, content_id, page, bbox).
    """
    import re
    import os
    chunks = []
    basename = os.path.basename(pdf_path)
    for page_num, figures in enumerate(detections):
        for idx, fig in enumerate(figures):
            caption = fig.get("caption") or fig.get("text") or ""
            bbox = fig.get("bbox")
            # Try to extract figure number from caption (e.g., 'Figure 2: ...')
            match = re.search(r"figure\s*(\d+)", caption, re.IGNORECASE)
            content_id = match.group(1) if match else str(idx + 1)
            chunk = {
                "id": f"{basename}-page{page_num}-figure{content_id}",
                "text": caption,
                "metadata": {
                    "content_type": "figure",
                    "content_id": content_id,
                    "page": page_num,
                    "bbox": bbox,
                    "source": basename
                }
            }
            chunks.append(chunk)
    return chunks
