"""
Prompt Templates for Phase 3.4.

This module provides templates for generating prompts for different query types
in a scientific RAG system, including system instructions, few-shot examples,
and query-specific formatting.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import logging
import re
from enum import Enum
from essentials.phase3_1.models import Chunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enumeration of query types for template selection."""
    GENERAL = "general"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    LITERATURE = "literature"
    SYNTHESIS = "synthesis"

class PromptTemplate:
    """Template for generating prompts with context and instructions."""
    
    def __init__(
        self,
        system_message: str,
        context_prefix: str = "Here is information to help answer the question:\n\n",
        query_prefix: str = "Question: ",
        answer_prefix: str = "Answer: ",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        context_format: str = "default"
    ):
        """Initialize the prompt template.
        
        Args:
            system_message: System message for the LLM
            context_prefix: Text to prefix the context with
            query_prefix: Text to prefix the query with
            answer_prefix: Text to prefix the answer with
            few_shot_examples: Optional list of few-shot examples
            context_format: Format for the context (default, scientific, etc.)
        """
        self.system_message = system_message
        self.context_prefix = context_prefix
        self.query_prefix = query_prefix
        self.answer_prefix = answer_prefix
        self.few_shot_examples = few_shot_examples or []
        self.context_format = context_format
    
    def format_context(self, context: str) -> str:
        """Format the context according to the specified format.
        
        Args:
            context: Context to format
            
        Returns:
            Formatted context
        """
        if not context:
            return ""
            
        if self.context_format == "scientific":
            # Add section breaks and citations in scientific format
            formatted = context.replace("[Source:", "\n\nSource:")
            formatted = formatted.replace("[Section:", "\n\nSection:")
            return formatted
        elif self.context_format == "compact":
            # More compact format with less whitespace
            formatted = context.replace("\n\n", "\n")
            return formatted
        else:
            # Default format - return as is
            return context
    
    def format_few_shot_examples(self) -> str:
        """Format the few-shot examples.
        
        Returns:
            Formatted few-shot examples string
        """
        if not self.few_shot_examples:
            return ""
            
        examples = []
        for example in self.few_shot_examples:
            query = example.get("query", "")
            answer = example.get("answer", "")
            context = example.get("context", "")
            
            example_parts = []
            
            # Add context if available
            if context:
                example_parts.append(f"{self.context_prefix}{context}")
                
            # Add query
            example_parts.append(f"{self.query_prefix}{query}")
            
            # Add answer
            example_parts.append(f"{self.answer_prefix}{answer}")
            
            # Add full example
            examples.append("\n\n".join(example_parts))
        
        return "\n\n---\n\n".join(examples)
    
    def create_prompt(
        self,
        query: str,
        context: str,
        include_few_shot: bool = True
    ) -> Dict[str, str]:
        """Create a prompt with system message, context, and query.
        
        Args:
            query: User query
            context: Context for the query
            include_few_shot: Whether to include few-shot examples
            
        Returns:
            Dictionary with system_message, prompt
        """
        formatted_context = self.format_context(context)
        prompt_parts = []
        
        # Add few-shot examples if requested
        if include_few_shot and self.few_shot_examples:
            prompt_parts.append(self.format_few_shot_examples())
            prompt_parts.append("---\n")
        
        # Add context if available
        if formatted_context:
            prompt_parts.append(f"{self.context_prefix}{formatted_context}")
        
        # Add query
        prompt_parts.append(f"{self.query_prefix}{query}")
        
        # Add answer prefix
        prompt_parts.append(self.answer_prefix)
        
        # Join all parts
        prompt = "\n\n".join(prompt_parts)
        
        return {
            "system_message": self.system_message,
            "prompt": prompt
        }


class PromptTemplateLibrary:
    """Library of prompt templates for different query types."""
    
    def __init__(self):
        """Initialize the prompt template library with default templates."""
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize the default templates."""
        # General template
        self.templates[QueryType.GENERAL] = PromptTemplate(
            system_message=(
                "You are a scientific assistant that provides accurate, factual information based on the "
                "provided context. Focus on answering the question directly using only the information "
                "in the context. If the context doesn't contain the answer, admit that you don't know. "
                "Do not make up information or rely on prior knowledge not present in the context."
            ),
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Journal of AI Research | Section: Introduction]\n"
                        "Recent advances in large language models (LLMs) have demonstrated impressive capabilities "
                        "in natural language understanding and generation. However, these models still face challenges "
                        "with factual accuracy and hallucination.\n\n"
                        "[Source: AI Conference Proceedings | Section: Methodology]\n"
                        "Retrieval-augmented generation (RAG) systems address these limitations by retrieving relevant "
                        "information from external knowledge sources and using it to ground the model's responses."
                    ),
                    "query": "What is the main benefit of retrieval-augmented generation systems?",
                    "answer": (
                        "According to the context, the main benefit of retrieval-augmented generation (RAG) systems "
                        "is that they address the limitations of large language models related to factual accuracy "
                        "and hallucination. They do this by retrieving relevant information from external knowledge "
                        "sources and using it to ground the model's responses."
                    )
                }
            ]
        )
        
        # Methodology template
        self.templates[QueryType.METHODOLOGY] = PromptTemplate(
            system_message=(
                "You are a scientific assistant specializing in research methodologies. When answering questions, "
                "focus on the specific methods, procedures, techniques, and experimental designs described in the "
                "context. Provide step-by-step explanations when appropriate. Be precise about the methodology "
                "details and limitations mentioned in the provided information."
            ),
            context_format="scientific",
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Nature Methods | Section: Experimental Procedure]\n"
                        "For protein extraction, cells were lysed in RIPA buffer supplemented with protease inhibitors "
                        "(1:100 dilution) and incubated on ice for 30 minutes. Lysates were centrifuged at 14,000g for "
                        "15 minutes at 4°C, and protein concentrations were determined using the Bradford assay.\n\n"
                        "[Source: Cell Biology Protocols | Section: Methods]\n"
                        "Western blotting was performed by loading 20μg of protein per sample onto 10% SDS-PAGE gels. "
                        "Following electrophoresis, proteins were transferred to PVDF membranes, blocked with 5% non-fat "
                        "milk, and incubated with primary antibodies overnight at 4°C."
                    ),
                    "query": "How was protein concentration measured in the experiment?",
                    "answer": (
                        "According to the context, protein concentration was measured using the Bradford assay. This was "
                        "done after the cells were lysed in RIPA buffer supplemented with protease inhibitors (1:100 dilution), "
                        "incubated on ice for 30 minutes, and centrifuged at 14,000g for 15 minutes at 4°C."
                    )
                }
            ]
        )
        
        # Results template
        self.templates[QueryType.RESULTS] = PromptTemplate(
            system_message=(
                "You are a scientific assistant specializing in analyzing and explaining research results. When answering "
                "questions, focus on the findings, data, statistics, and conclusions presented in the context. Be precise "
                "about quantitative results, statistical significance, and the interpretation of findings. Distinguish "
                "between primary results and secondary observations when possible."
            ),
            context_format="scientific",
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Journal of Clinical Medicine | Section: Results]\n"
                        "The treatment group (n=45) showed a significant reduction in LDL cholesterol levels compared to "
                        "the placebo group (n=42) after 12 weeks (mean difference: -38.5 mg/dL, 95% CI: -45.2 to -31.8, "
                        "p<0.001). Additionally, HDL cholesterol increased by 4.2 mg/dL (95% CI: 2.1 to 6.3, p=0.002) in "
                        "the treatment group.\n\n"
                        "[Source: Clinical Trial Report | Section: Adverse Events]\n"
                        "Mild gastrointestinal symptoms were reported by 15% of participants in the treatment group versus "
                        "7% in the placebo group. No serious adverse events were attributed to the study medication."
                    ),
                    "query": "What was the effect of the treatment on LDL cholesterol levels?",
                    "answer": (
                        "According to the results in the context, the treatment group (n=45) showed a significant reduction "
                        "in LDL cholesterol levels compared to the placebo group (n=42) after 12 weeks. The mean difference "
                        "was -38.5 mg/dL (95% CI: -45.2 to -31.8) with a p-value less than 0.001, indicating statistical "
                        "significance."
                    )
                }
            ]
        )
        
        # Comparison template
        self.templates[QueryType.COMPARISON] = PromptTemplate(
            system_message=(
                "You are a scientific assistant specializing in comparative analysis. When answering questions, focus on "
                "comparing and contrasting different methods, findings, or theories presented in the context. Highlight "
                "key similarities and differences, advantages and limitations, and provide balanced evaluations based on "
                "the evidence provided."
            ),
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Machine Learning Review | Section: Algorithms]\n"
                        "Random Forest algorithms create multiple decision trees and merge their predictions, making them "
                        "robust against overfitting and capable of handling high-dimensional data. However, they can be "
                        "computationally intensive for very large datasets.\n\n"
                        "[Source: Machine Learning Review | Section: Algorithms]\n"
                        "Support Vector Machines (SVMs) excel at finding optimal decision boundaries in high-dimensional "
                        "spaces and are effective with clear margins of separation. They struggle with large datasets and "
                        "require careful kernel selection and parameter tuning."
                    ),
                    "query": "Compare Random Forest and SVM algorithms for machine learning.",
                    "answer": (
                        "Based on the context, Random Forest and Support Vector Machine (SVM) algorithms have several key "
                        "differences:\n\n"
                        "Random Forest:\n"
                        "- Creates multiple decision trees and merges predictions\n"
                        "- Robust against overfitting\n"
                        "- Handles high-dimensional data well\n"
                        "- Can be computationally intensive for very large datasets\n\n"
                        "Support Vector Machines (SVMs):\n"
                        "- Excel at finding optimal decision boundaries in high-dimensional spaces\n"
                        "- Effective with clear margins of separation\n"
                        "- Struggle with large datasets\n"
                        "- Require careful kernel selection and parameter tuning\n\n"
                        "Both algorithms can handle high-dimensional data, but they differ in computational requirements "
                        "and their approach to model building."
                    )
                }
            ]
        )
        
        # Definition template
        self.templates[QueryType.DEFINITION] = PromptTemplate(
            system_message=(
                "You are a scientific assistant specializing in technical definitions and explanations. When answering "
                "questions, focus on providing clear, precise definitions of terms, concepts, and phenomena based on the "
                "provided context. Include key characteristics, classifications, and relationships to other concepts "
                "when relevant."
            ),
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Biochemistry Textbook | Section: Enzymes]\n"
                        "Allosteric enzymes are proteins that change their conformational structure upon binding with an "
                        "effector molecule at a site other than the active site, resulting in altered catalytic activity. "
                        "This regulation can be positive (activating) or negative (inhibiting) and allows for sophisticated "
                        "control of metabolic pathways.\n\n"
                        "[Source: Cell Biology Journal | Section: Metabolic Regulation]\n"
                        "A classic example of allosteric regulation is seen in phosphofructokinase, a key regulatory enzyme "
                        "in glycolysis that is inhibited by ATP and activated by AMP, allowing the pathway to respond to the "
                        "cell's energy status."
                    ),
                    "query": "What are allosteric enzymes?",
                    "answer": (
                        "According to the context, allosteric enzymes are proteins that change their conformational structure "
                        "when they bind with an effector molecule at a site other than the active site. This binding results "
                        "in altered catalytic activity. The regulation can be either positive (activating) or negative "
                        "(inhibiting), which allows for sophisticated control of metabolic pathways. Phosphofructokinase is "
                        "provided as a classic example of an enzyme with allosteric regulation; it's a key regulatory enzyme "
                        "in glycolysis that is inhibited by ATP and activated by AMP, allowing the pathway to respond to the "
                        "cell's energy status."
                    )
                }
            ]
        )
        
        # Literature template
        self.templates[QueryType.LITERATURE] = PromptTemplate(
            system_message=(
                "You are a scientific assistant specializing in literature review and research synthesis. When answering "
                "questions, focus on summarizing the current state of knowledge, key studies, competing theories, and "
                "research gaps based on the provided context. Acknowledge areas of consensus and controversy, and cite "
                "sources appropriately."
            ),
            context_format="scientific",
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Neuropsychology Review 2020 | Section: Literature Review]\n"
                        "Smith et al. (2018) conducted a meta-analysis of 42 studies examining the relationship between "
                        "meditation practice and cognitive function in older adults. They found consistent evidence for "
                        "improvements in attention and working memory (effect size d=0.35, p<0.001).\n\n"
                        "[Source: Journal of Cognitive Neuroscience 2019 | Section: Discussion]\n"
                        "In contrast to Smith's findings, Jones and Brown (2019) reported minimal cognitive benefits in "
                        "their randomized controlled trial of brief meditation training (d=0.12, p=0.08). They suggested "
                        "that longer intervention periods may be necessary for measurable effects."
                    ),
                    "query": "What does the literature say about meditation's effects on cognitive function?",
                    "answer": (
                        "Based on the provided context, the literature on meditation's effects on cognitive function shows "
                        "some mixed results:\n\n"
                        "Smith et al. (2018) conducted a meta-analysis of 42 studies examining meditation practice and "
                        "cognitive function in older adults. They found consistent evidence for improvements in attention "
                        "and working memory with a significant effect size (d=0.35, p<0.001).\n\n"
                        "However, Jones and Brown (2019) reported minimal cognitive benefits in their randomized controlled "
                        "trial of brief meditation training, with a smaller, non-significant effect size (d=0.12, p=0.08). "
                        "They suggested that longer intervention periods may be necessary to achieve measurable effects.\n\n"
                        "This suggests that while there is some evidence supporting cognitive benefits from meditation, the "
                        "duration and possibly other characteristics of the meditation practice may influence the outcomes."
                    )
                }
            ]
        )
        
        # Synthesis template
        self.templates[QueryType.SYNTHESIS] = PromptTemplate(
            system_message=(
                "You are a scientific assistant specializing in synthesizing complex information from multiple sources. "
                "When answering questions, focus on integrating information across different sources in the context to "
                "provide comprehensive, well-rounded answers. Identify patterns, connections, and broader implications "
                "while maintaining accuracy to each source."
            ),
            few_shot_examples=[
                {
                    "context": (
                        "[Source: Environmental Science Journal | Section: Climate Impacts]\n"
                        "Rising sea levels are projected to increase coastal flooding events by 25-50% in vulnerable regions "
                        "by 2050, affecting approximately 300 million people living in low-elevation coastal zones.\n\n"
                        "[Source: Urban Planning Review | Section: Adaptation Strategies]\n"
                        "Coastal adaptation strategies include hard infrastructure (sea walls, levees), soft approaches "
                        "(beach nourishment, wetland restoration), and managed retreat. Cost-benefit analyses suggest "
                        "combined approaches are most effective.\n\n"
                        "[Source: Economic Policy Report | Section: Financial Implications]\n"
                        "The estimated global cost of coastal protection is $71-111 billion annually by 2050, but averted "
                        "damages could reach $1 trillion per year."
                    ),
                    "query": "How should coastal communities respond to rising sea levels?",
                    "answer": (
                        "Based on synthesizing the provided context, coastal communities facing rising sea levels should "
                        "consider comprehensive response strategies:\n\n"
                        "The threat is significant: Sea levels are projected to increase coastal flooding events by 25-50% "
                        "in vulnerable regions by 2050, potentially affecting around 300 million people living in low-elevation "
                        "coastal zones.\n\n"
                        "Available adaptation strategies include:\n"
                        "- Hard infrastructure approaches like sea walls and levees\n"
                        "- Soft approaches such as beach nourishment and wetland restoration\n"
                        "- Managed retreat from highly vulnerable areas\n\n"
                        "From an economic perspective, the estimated global cost of coastal protection ranges from $71-111 "
                        "billion annually by 2050. However, these investments could help avert damages of up to $1 trillion "
                        "per year, suggesting a strong economic case for adaptation measures.\n\n"
                        "Cost-benefit analyses indicate that combined approaches—integrating multiple strategies rather than "
                        "relying on a single method—are most effective for coastal protection."
                    )
                }
            ]
        )
    
    def get_template(self, query_type: Union[str, QueryType]) -> PromptTemplate:
        """Get a template for a query type.
        
        Args:
            query_type: Query type (string or enum)
            
        Returns:
            Prompt template for the query type
        """
        # Convert string to enum if needed
        if isinstance(query_type, str):
            try:
                query_type = QueryType(query_type.lower())
            except ValueError:
                logger.warning(f"Unknown query type: {query_type}, using GENERAL")
                query_type = QueryType.GENERAL
        
        # Return template or default
        return self.templates.get(query_type, self.templates[QueryType.GENERAL])
    
    def add_template(self, query_type: Union[str, QueryType], template: PromptTemplate):
        """Add a new template or replace an existing one.
        
        Args:
            query_type: Query type (string or enum)
            template: Prompt template
        """
        # Convert string to enum if needed
        if isinstance(query_type, str):
            try:
                query_type = QueryType(query_type.lower())
            except ValueError:
                try:
                    # Try to create a new enum value
                    QueryType(query_type.lower())
                    query_type = QueryType(query_type.lower())
                except:
                    logger.warning(f"Invalid query type: {query_type}, cannot add template")
                    return
        
        # Add or replace template
        self.templates[query_type] = template
        logger.info(f"Added template for query type: {query_type.name}")
    
    def detect_query_type(self, query: str) -> QueryType:
        """Detect the type of a query using simple heuristics.
        
        Args:
            query: Query text
            
        Returns:
            Detected query type
        """
        query = query.lower()
        
        # Check for methodology queries
        if any(kw in query for kw in ["how to", "method", "procedure", "protocol", "technique", 
                                      "approach", "steps", "how do", "process"]):
            return QueryType.METHODOLOGY
        
        # Check for results queries
        if any(kw in query for kw in ["result", "finding", "outcome", "data show", "demonstrate", 
                                     "evidence", "statistical", "significant"]):
            return QueryType.RESULTS
        
        # Check for comparison queries
        if any(kw in query for kw in ["compare", "contrast", "difference", "similarity", "versus", 
                                     "better", "worse", "advantage", "disadvantage"]):
            return QueryType.COMPARISON
        
        # Check for definition queries
        if any(kw in query for kw in ["what is", "define", "meaning", "definition", "explain", 
                                     "describe", "concept"]):
            return QueryType.DEFINITION
        
        # Check for literature queries
        if any(kw in query for kw in ["literature", "research", "study", "publication", "paper", 
                                     "review", "meta-analysis", "published"]):
            return QueryType.LITERATURE
        
        # Check for synthesis queries
        if any(kw in query for kw in ["synthesize", "integrate", "combine", "overall", "big picture", 
                                     "holistic", "comprehensive", "summarize"]):
            return QueryType.SYNTHESIS
        
        # Default to general
        return QueryType.GENERAL
    
    def create_prompt(
        self,
        query: str,
        context: str,
        query_type: Optional[Union[str, QueryType]] = None,
        include_few_shot: bool = True
    ) -> Dict[str, str]:
        """Create a prompt for a query with appropriate template.
        
        Args:
            query: User query
            context: Context for the query
            query_type: Optional query type (if None, auto-detect)
            include_few_shot: Whether to include few-shot examples
            
        Returns:
            Dictionary with system_message, prompt
        """
        # Auto-detect query type if not provided
        if query_type is None:
            query_type = self.detect_query_type(query)
        
        # Get template and create prompt
        template = self.get_template(query_type)
        return template.create_prompt(query, context, include_few_shot) 