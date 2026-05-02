"""Link in-text citations to their reference entries."""

import re
from typing import Dict, List, Optional, Tuple, Any

from ...utils.logging import get_logger
from ...utils.types import Document, Citation, Reference, LinkedCitation

logger = get_logger(__name__)


class CitationLinker:
    """Link in-text citations to their reference entries."""
    
    def __init__(self, **kwargs):
        """Initialize the citation linker.
        
        Args:
            **kwargs: Additional configuration options
        """
        # Configure linker options
        self.fuzzy_matching = kwargs.get("fuzzy_matching", True)
        self.min_confidence = kwargs.get("min_confidence", 0.6)
        
        logger.debug(
            f"Initialized CitationLinker with fuzzy_matching={self.fuzzy_matching}"
        )
    
    def link_citations(
        self,
        citations: List[Citation],
        references: List[Reference]
    ) -> List[LinkedCitation]:
        """Link citations to their reference entries.
        
        Args:
            citations: List of citations
            references: List of references
            
        Returns:
            List[LinkedCitation]: List of linked citations
        """
        if not citations or not references:
            return []
        
        linked_citations = []
        
        # Create a reference lookup dictionary
        ref_lookup = {ref.ref_id: ref for ref in references}
        
        # Link each citation
        for citation in citations:
            linked_refs = []
            confidence = 0.0
            
            # Try to link by ID for numeric citations
            if citation.citation_type == "numeric":
                # Try to find each referenced ID
                for ref_id in citation.ref_ids:
                    if ref_id in ref_lookup:
                        linked_refs.append(ref_lookup[ref_id])
                        confidence = max(confidence, 0.95)
            
            # Try to link by author/year for author-year citations
            elif citation.citation_type in ["author_year", "multiple_author_year"]:
                # Try exact match first by combining author and year
                for ref_id in citation.ref_ids:
                    if ref_id in ref_lookup:
                        linked_refs.append(ref_lookup[ref_id])
                        confidence = max(confidence, 0.9)
                
                # If no exact match, try fuzzy matching if enabled
                if not linked_refs and self.fuzzy_matching and citation.authors:
                    # Try matching by author and year
                    for ref in references:
                        # Check for author match
                        author_match = False
                        for citation_author in citation.authors:
                            # Get last name of citation author
                            citation_lastname = citation_author.split(",")[0].strip() if "," in citation_author else citation_author.strip()
                            
                            # Check if author appears in reference
                            if any(citation_lastname.lower() in ref_author.lower() for ref_author in ref.authors):
                                author_match = True
                                break
                        
                        # Check for year match if we have author match
                        if author_match and citation.year and ref.year and citation.year == ref.year:
                            linked_refs.append(ref)
                            confidence = max(confidence, 0.8)
            
            # Create linked citation
            if linked_refs:
                linked_citation = LinkedCitation(
                    citation=citation,
                    references=linked_refs,
                    confidence=confidence
                )
                linked_citations.append(linked_citation)
        
        logger.info(f"Linked {len(linked_citations)} citations to references")
        return linked_citations
    
    def process_document(
        self,
        document: Document,
        citations: Optional[List[Citation]] = None,
        references: Optional[List[Reference]] = None
    ) -> Dict[str, Any]:
        """Extract, parse, and link citations and references from a document.
        
        Args:
            document: The document to process
            citations: Pre-extracted citations (optional)
            references: Pre-extracted references (optional)
            
        Returns:
            Dict[str, Any]: Dictionary with citations, references, and linked citations
        """
        from .citation_extractor import CitationExtractor
        from .reference_parser import ReferenceParser
        
        # Extract citations if not provided
        if citations is None:
            citation_extractor = CitationExtractor()
            section_citations = citation_extractor.extract_document_citations(document)
            
            # Flatten citations from all sections
            citations = []
            for section_name, section_cites in section_citations.items():
                citations.extend(section_cites)
        
        # Extract references if not provided
        if references is None:
            reference_parser = ReferenceParser()
            references = reference_parser.extract_references_section(document)
        
        # Link citations to references
        linked_citations = self.link_citations(citations, references)
        
        # Return results
        return {
            "citations": citations,
            "references": references,
            "linked_citations": linked_citations,
            "citation_count": len(citations),
            "reference_count": len(references),
            "linked_count": len(linked_citations),
            "unlinked_count": len(citations) - len(linked_citations)
        }
    
    def compute_citation_metrics(
        self,
        linked_citations: List[LinkedCitation]
    ) -> Dict[str, Any]:
        """Compute citation metrics from linked citations.
        
        Args:
            linked_citations: List of linked citations
            
        Returns:
            Dict[str, Any]: Citation metrics
        """
        if not linked_citations:
            return {
                "total_citations": 0,
                "unique_references_cited": 0,
                "citation_distribution": {}
            }
        
        # Count citations per reference
        ref_counts = {}
        
        for linked in linked_citations:
            for ref in linked.references:
                ref_id = ref.ref_id
                ref_counts[ref_id] = ref_counts.get(ref_id, 0) + 1
        
        # Compute metrics
        metrics = {
            "total_citations": len(linked_citations),
            "unique_references_cited": len(ref_counts),
            "citation_distribution": ref_counts,
            "max_citations": max(ref_counts.values()) if ref_counts else 0,
            "avg_citations_per_reference": sum(ref_counts.values()) / len(ref_counts) if ref_counts else 0
        }
        
        # Most cited references
        most_cited = sorted(ref_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics["most_cited_references"] = most_cited
        
        return metrics 