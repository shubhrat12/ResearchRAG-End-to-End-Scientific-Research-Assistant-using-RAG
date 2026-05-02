from typing import List, Dict
import networkx as nx
from community import community_louvain
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO)

class FieldSpecificSearchOptimizer:
    def __init__(self, field: str):
        """Initialize the optimizer with a specific field."""
        self.field = field
        self.field_terms = self._get_field_terms()

    def _get_field_terms(self) -> List[str]:
        """Retrieve field-specific terms for optimizing search queries."""
        # Placeholder for field-specific terms
        field_terms_dict: Dict[str, List[str]] = {
            'biology': ['gene', 'protein', 'cell'],
            'computer science': ['algorithm', 'data structure', 'network'],
            # Add more fields and terms as needed
        }
        return field_terms_dict.get(self.field, [])

    def optimize_query(self, query: str) -> str:
        """Optimize the search query by adding field-specific terms."""
        optimized_query = query + ' ' + ' '.join(self.field_terms)
        logging.info(f"Optimized Query: {optimized_query}")
        return optimized_query

    def diversify_results(self, results: List[str], max_per_category: int = 2) -> List[str]:
        """Diversify search results by limiting the number of results per category."""
        # Placeholder for categorization logic
        categorized_results: Dict[str, List[str]] = {
            'topic1': [],
            'topic2': [],
            # Add more categories as needed
        }

        # Example categorization (this should be replaced with actual logic)
        for result in results:
            if 'gene' in result:
                categorized_results['topic1'].append(result)
            else:
                categorized_results['topic2'].append(result)

        diversified_results = []
        for category, items in categorized_results.items():
            diversified_results.extend(items[:max_per_category])

        logging.info(f"Diversified Results: {diversified_results}")
        return diversified_results

    def calculate_relevance_scores(self, query: str, results: List[str]) -> Dict[str, float]:
        """Calculate relevance scores for each result based on similarity to the query."""
        query_terms = set(query.lower().split())
        relevance_scores = {}
        if not query_terms:
            logging.warning("Query terms are empty, returning empty relevance scores.")
            return relevance_scores
        
        for result in results:
            result_terms = set(result.lower().split())
            common_terms = query_terms.intersection(result_terms)
            relevance_scores[result] = len(common_terms) / len(query_terms)
        logging.info(f"Relevance Scores: {relevance_scores}")
        return relevance_scores

    def calculate_recency_weighted_scores(self, relevance_scores: Dict[str, float], publication_dates: Dict[str, str]) -> Dict[str, float]:
        """Adjust relevance scores based on recency using publication dates."""
        recency_weighted_scores = {}
        current_date = datetime.now()
        for result, relevance_score in relevance_scores.items():
            pub_date = datetime.strptime(publication_dates[result], "%Y-%m-%d")
            days_since_publication = (current_date - pub_date).days
            recency_score = 1 / (1 + days_since_publication / 365)  # Decay factor
            recency_weighted_scores[result] = relevance_score * recency_score
        logging.info(f"Recency Weighted Scores: {recency_weighted_scores}")
        return recency_weighted_scores

    def calculate_citation_influenced_scores(self, recency_weighted_scores: Dict[str, float], citation_counts: Dict[str, int]) -> Dict[str, float]:
        """Adjust scores based on citation counts."""
        citation_influenced_scores = {}
        max_citations = max(citation_counts.values()) if citation_counts else 1
        for result, recency_weighted_score in recency_weighted_scores.items():
            citation_score = citation_counts.get(result, 0) / max_citations
            citation_influenced_scores[result] = recency_weighted_score * (1 + citation_score)
        logging.info(f"Citation Influenced Scores: {citation_influenced_scores}")
        return citation_influenced_scores

    def promote_diversity(self, scores: Dict[str, float], categories: Dict[str, str], max_per_category: int = 2) -> List[str]:
        """Promote diversity by ensuring a balanced representation of categories in the final ranking."""
        categorized_results: Dict[str, List[str]] = {}
        for result, category in categories.items():
            if category not in categorized_results:
                categorized_results[category] = []
            categorized_results[category].append(result)

        # Sort results within each category by score
        for category, results in categorized_results.items():
            results.sort(key=lambda x: scores[x], reverse=True)

        # Collect top results from each category
        diverse_results = []
        for category, results in categorized_results.items():
            diverse_results.extend(results[:max_per_category])

        # Sort the final list by score
        diverse_results.sort(key=lambda x: scores[x], reverse=True)
        logging.info(f"Diverse Results: {diverse_results}")
        return diverse_results

class CitationNetwork:
    def __init__(self):
        """Initialize an empty citation network."""
        self.graph = nx.DiGraph()

    def add_paper(self, paper_id: str, title: str):
        """Add a paper to the citation network."""
        self.graph.add_node(paper_id, title=title)

    def add_citation(self, citing_paper_id: str, cited_paper_id: str):
        """Add a citation between two papers."""
        self.graph.add_edge(citing_paper_id, cited_paper_id)

    def get_network_info(self):
        """Return basic information about the citation network."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        logging.info(f"Citation Network Info: Nodes: {num_nodes}, Edges: {num_edges}")
        return f"Nodes: {num_nodes}, Edges: {num_edges}"

    def calculate_citation_count(self) -> Dict[str, int]:
        """Calculate the citation count for each paper."""
        logging.info(f"Citation Counts: {dict(self.graph.in_degree())}")
        return dict(self.graph.in_degree())

    def calculate_pagerank(self) -> Dict[str, float]:
        """Calculate the PageRank for each paper."""
        logging.info(f"PageRank: {nx.pagerank(self.graph)}")
        return nx.pagerank(self.graph)

    def detect_communities(self) -> Dict[int, List[str]]:
        """Detect communities within the citation network using the Louvain method."""
        partition = community_louvain.best_partition(self.graph.to_undirected())
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        logging.info(f"Communities: {communities}")
        return communities

    def export_to_graphml(self, file_path: str):
        """Export the citation network to a GraphML file for visualization."""
        nx.write_graphml(self.graph, file_path)
        logging.info("Citation network exported to citation_network.graphml")

class BatchDownloadManager:
    def __init__(self, download_dir: str):
        """Initialize the batch download manager with a directory to save files."""
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_files(self, urls: List[str]) -> List[str]:
        """Download files from a list of URLs and save them to the download directory."""
        downloaded_files = []
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                file_name = url.split('/')[-1]
                file_path = f"{self.download_dir}/{file_name}"
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                downloaded_files.append(file_path)
            except requests.RequestException as e:
                logging.error(f"Failed to download {url}: {e}")
        logging.info(f"Downloaded Files: {downloaded_files}")
        return downloaded_files

class ParallelProcessor:
    def __init__(self, max_workers: int = 4):
        """Initialize the parallel processor with a specified number of workers."""
        self.max_workers = max_workers

    def process_results(self, results: List[str], process_function) -> List:
        """Process search results in parallel using the provided processing function."""
        processed_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_result = {executor.submit(process_function, result): result for result in results}
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                try:
                    processed_result = future.result()
                    processed_results.append(processed_result)
                except Exception as e:
                    logging.error(f"Error processing {result}: {e}")
        logging.info(f"Processed Results: {processed_results}")
        return processed_results

class FullTextRetriever:
    def __init__(self, threshold: float = 0.5, keywords: List[str] = None):
        """Initialize the full-text retriever with a relevance threshold and optional keywords."""
        self.threshold = threshold
        self.keywords = keywords if keywords else []

    def retrieve_documents(self, scores: Dict[str, float], documents: Dict[str, str]) -> List[str]:
        """Retrieve documents that meet the relevance threshold and contain specified keywords."""
        retrieved_documents = []
        for doc_id, score in scores.items():
            if score >= self.threshold:
                document_text = documents.get(doc_id, "")
                if any(keyword.lower() in document_text.lower() for keyword in self.keywords):
                    retrieved_documents.append(doc_id)
        logging.info(f"Retrieved Documents: {retrieved_documents}")
        return retrieved_documents

class ResultSummarizer:
    def __init__(self, num_sentences: int = 2):
        """Initialize the result summarizer with the number of sentences for the summary."""
        self.num_sentences = num_sentences

    def summarize(self, document: str) -> str:
        """Generate a summary of the document by extracting key sentences."""
        sentences = document.split('. ')
        vectorizer = TfidfVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()
        sentence_scores = vectors.sum(axis=1)
        ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-self.num_sentences:][::-1]]
        summary = '. '.join(ranked_sentences)
        logging.info(f"Summary: {summary}")
        return summary

class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"
    HEADERS = {'User-Agent': 'Langchain-Scientific-NLP-Client/1.0'}

    def __init__(self, fields: list = None):
        self.fields = fields if fields else ["title", "abstract", "citationCount"]

    def get_metadata(self, paper_id: str) -> dict:
        url = f"{self.BASE_URL}{paper_id}"
        params = {"fields": ",".join(self.fields)}
        try:
            response = requests.get(url, params=params, headers=self.HEADERS)
            if response.status_code == 200:
                logging.info(f"Successfully fetched metadata for paper ID: {paper_id}")
                return response.json()
            else:
                logging.error(f"Failed to fetch metadata for paper ID: {paper_id}, Status Code: {response.status_code}")
                response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Error fetching metadata for paper ID: {paper_id}: {e}")
            raise
        finally:
            time.sleep(3)  # Respect rate limit

def search_semantic_scholar(query: str, limit: int = 5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,citationCount,url"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        logging.info(f"Fetched {len(response.json()['data'])} papers from Semantic Scholar for query: {query}")
        return response.json()["data"]
    except requests.RequestException as e:
        logging.error(f"Error fetching data from Semantic Scholar: {e}")
        return []

if __name__ == "__main__":
    # Example usage for result diversification
    optimizer = FieldSpecificSearchOptimizer(field='biology')
    results = ["gene therapy advances", "protein folding studies", "cellular mechanisms", "network algorithms"]
    diversified_results = optimizer.diversify_results(results)
    
    # Example usage for citation network
    citation_network = CitationNetwork()
    citation_network.add_paper("P1", "Deep Learning for NLP")
    citation_network.add_paper("P2", "Advances in Machine Learning")
    citation_network.add_paper("P3", "Neural Networks")
    citation_network.add_citation("P1", "P2")
    citation_network.add_citation("P2", "P3")
    print("Citation Network Info:", citation_network.get_network_info())
    print("Citation Counts:", citation_network.calculate_citation_count())
    print("PageRank:", citation_network.calculate_pagerank())
    
    # Example usage for community detection
    print("Communities:", citation_network.detect_communities())
    
    # Example usage for visualization preparation
    citation_network.export_to_graphml("citation_network.graphml")
    print("Citation network exported to citation_network.graphml")
    
    # Example usage for relevance scoring
    query = "COVID-19 vaccine"
    results = ["COVID-19 vaccine development", "gene therapy advances", "vaccine distribution"]
    relevance_scores = optimizer.calculate_relevance_scores(query, results)
    print("Relevance Scores:", relevance_scores)
    
    # Example usage for recency weighting
    publication_dates = {'COVID-19 vaccine development': '2023-10-01', 'gene therapy advances': '2020-05-15', 'vaccine distribution': '2022-08-20'}
    recency_weighted_scores = optimizer.calculate_recency_weighted_scores(relevance_scores, publication_dates)
    print("Recency Weighted Scores:", recency_weighted_scores)
    
    # Example usage for citation count influence
    citation_counts = {'COVID-19 vaccine development': 50, 'gene therapy advances': 10, 'vaccine distribution': 30}
    citation_influenced_scores = optimizer.calculate_citation_influenced_scores(recency_weighted_scores, citation_counts)
    print("Citation Influenced Scores:", citation_influenced_scores)
    
    # Example usage for diversity promotion
    scores = {'COVID-19 vaccine development': 0.758, 'gene therapy advances': 0.0, 'vaccine distribution': 0.2128}
    categories = {'COVID-19 vaccine development': 'health', 'gene therapy advances': 'genetics', 'vaccine distribution': 'health'}
    diverse_results = optimizer.promote_diversity(scores, categories)
    print("Diverse Results:", diverse_results)
    
    # Example usage for batch download manager
    download_manager = BatchDownloadManager(download_dir='downloads')
    urls = [
        "https://hal.science/hal-04206682/document",
        "https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf"
    ]
    downloaded_files = download_manager.download_files(urls)
    print("Downloaded Files:", downloaded_files)
    
    # Example usage for parallel processing
    def example_process_function(result):
        return result.upper()

    processor = ParallelProcessor(max_workers=2)
    results = ["result1", "result2", "result3"]
    processed_results = processor.process_results(results, example_process_function)
    print("Processed Results:", processed_results)
    
    # Example usage for selective full-text retrieval
    retriever = FullTextRetriever(threshold=0.5, keywords=["vaccine", "COVID-19"])
    scores = {'doc1': 0.7, 'doc2': 0.4, 'doc3': 0.6}
    documents = {'doc1': "This document discusses the COVID-19 vaccine.", 'doc2': "This is about gene therapy.", 'doc3': "Vaccine distribution is crucial."}
    retrieved_docs = retriever.retrieve_documents(scores, documents)
    print("Retrieved Documents:", retrieved_docs)
    
    # Example usage for search result summarization
    summarizer = ResultSummarizer(num_sentences=2)
    document = "The COVID-19 vaccine is effective. It has been distributed worldwide. Many people have received it. The vaccine helps prevent severe illness."
    summary = summarizer.summarize(document)
    print("Summary:", summary)

    # Example usage of the new function
    query = "COVID-19 vaccine"
    papers = search_semantic_scholar(query)
    
    # Initialize once outside the loop
    optimizer = FieldSpecificSearchOptimizer(field='biology')
    summarizer = ResultSummarizer(num_sentences=2)
    
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        citation_count = paper.get("citationCount", 0)
        url = paper.get("url", "")
        
        # Use title and abstract in FieldSpecificSearchOptimizer
        relevance_scores = optimizer.calculate_relevance_scores(query, [title])
        diversified_results = optimizer.diversify_results([title])
        
        # Use abstract in ResultSummarizer
        summary = summarizer.summarize(abstract)
        
        # Use citation count in calculate_citation_influenced_scores
        citation_influenced_scores = optimizer.calculate_citation_influenced_scores(relevance_scores, {title: citation_count})
        
        logging.info(f"Title: {title}")
        logging.info(f"Abstract: {abstract}")
        logging.info(f"Citation Count: {citation_count}")
        logging.info(f"URL: {url}")
        logging.info(f"Summary: {summary}")
        logging.info(f"Relevance Scores: {relevance_scores}")
        logging.info(f"Diversified Results: {diversified_results}")
        logging.info(f"Citation Influenced Scores: {citation_influenced_scores}")

    # Example usage of the SemanticScholarClient
    client = SemanticScholarClient()
    paper_id = "10.1109/5.771073"
    try:
        metadata = client.get_metadata(paper_id)
        logging.info(f"Metadata: {metadata}")
    except Exception as e:
        logging.error(f"An error occurred: {e}") 