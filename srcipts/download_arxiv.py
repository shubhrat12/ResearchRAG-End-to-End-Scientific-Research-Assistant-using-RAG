import os
import time
import json
from tqdm import tqdm

try:
    import arxiv
except ImportError:
    print("Installing arxiv package...")
    import subprocess
    subprocess.check_call(["pip", "install", "arxiv"])
    import arxiv

# Create directories
os.makedirs("data/training/arxiv_papers", exist_ok=True)
os.makedirs("data/training/arxiv_papers/metadata", exist_ok=True)

# Define search query (Computer Science, AI, ML)
search = arxiv.Search(
    query="cat:cs.AI OR cat:cs.CL OR cat:cs.CV",
    max_results=5000,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Download metadata
print("Downloading metadata for 5000 papers from arXiv...")
papers = []
for i, result in tqdm(enumerate(search.results()), desc="Downloading", total=5000):
    paper = {
        "paper_id": result.entry_id.split('/')[-1],
        "title": result.title,
        "abstract": result.summary,
        "authors": [author.name for author in result.authors],
        "categories": result.categories,
        "published": result.published.strftime("%Y-%m-%d"),
        "pdf_url": result.pdf_url
    }
    papers.append(paper)
    
    # Save in batches of 1000
    if (i+1) % 1000 == 0 or i == 4999:
        batch = (i+1) // 1000
        with open(f"data/training/arxiv_papers/metadata/batch_{batch}.json", 'w') as f:
            json.dump(papers[(batch-1)*1000:batch*1000], f, indent=2)
        
    # Be kind to arXiv API
    if i % 100 == 0 and i > 0:
        time.sleep(1)

print(f"Successfully downloaded metadata for {len(papers)} papers")
