import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO)

def summarize_sections(classified_sections):
    logging.info("Summarizing sections")
    summaries = []
    for section in classified_sections:
        summary = section['text'][:100]  # Placeholder for actual summarization logic
        summaries.append(summary)
    return summaries 