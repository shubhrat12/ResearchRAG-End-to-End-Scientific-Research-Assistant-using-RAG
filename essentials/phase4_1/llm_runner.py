import os
os.environ.setdefault("LLAMA_LOG_LEVEL", "40")   # 40 = ERROR
import llama_cpp  # now imports with log level already set
from pathlib import Path

from llama_cpp import Llama
from essentials.phase4_1.llm_config import MODEL_PATH, QUANTIZATION_TYPE, CONTEXT_WINDOW, TEMPERATURE
import logging
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_PROMPT = """
You are an expert scientific assistant. Use ONLY the provided context to answer.

⚠️  If the user's question refers to any figures, tables, or equations:
– Identify each by its number (e.g., "Figure 2").
– Give **one concise sentence** describing what it depicts, grounded strictly in the context.
– If the context lacks that item, respond: "The answer is not available in the provided document."

Do NOT invent content. Prefer clarity over verbosity.
"""

class LLMRunner:
    def __init__(self):
        try:
            self.model = Llama(model_path=MODEL_PATH, n_ctx=int(CONTEXT_WINDOW), temperature=float(TEMPERATURE), n_threads=4)
            logging.info("✅ Mistral model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def generate(self, prompt: str, context: str = '', title: str = '', authors: list = None, references: list = None) -> str:
        if not self.model:
            return "Model not loaded."
        authors = authors or []
        references = references or []
        # Deduplicate and clean references (up to 5)
        ref_key = lambda ref: (ref.get('title', '').strip().lower(), tuple(sorted(ref.get('authors', []))), str(ref.get('year', '')))
        seen = set()
        cleaned_refs = []
        for ref in references:
            key = ref_key(ref)
            if key in seen:
                continue
            seen.add(key)
            if not ref.get('title', '').strip():
                continue
            cleaned_refs.append(ref)
            if len(cleaned_refs) == 5:
                break
        truncated = len(references) > 5
        # Format references
        def format_reference(ref):
            title = ref.get('title', 'Unknown Title')
            authors = ', '.join(ref.get('authors', []))
            year = ref.get('year', 'Unknown Year')
            journal = ref.get('journal', '')
            doi = ref.get('doi', '')
            formatted = f"*{title}* by {authors} ({year})"
            if journal:
                formatted += f". {journal}"
            if doi:
                formatted += f". {doi}"
            return formatted
        formatted_references = [format_reference(ref) for ref in cleaned_refs]
        # Log and handle empty references
        if not formatted_references:
            formatted_references = ["No reliable references were found in the document."]
        elif truncated:
            formatted_references.append("... (references truncated for token limit)")
        logging.info(f"References for prompt: {formatted_references}")
        # Build references section for prompt
        references_section = '\n'.join(f"- {ref}" for ref in formatted_references)
        # Build prompt template
        prompt_template = f"""
        {SYSTEM_PROMPT}

        📘 Paper Context:
        ---
        {context}
        ---

        📚 References:
        {references_section}

        Now answer:

        Q: {prompt}
        A:
        """
        try:
            logging.info(f"Final Prompt: {prompt_template}")
            response = self.model.create_chat_completion(messages=[{"role": "user", "content": prompt_template}], max_tokens=1024, stream=False)
            logging.info(f"Raw Response: {response}")
            answer = response.get('choices', [{}])[0].get('message', {}).get('content', "")
            # If answer ends with a colon and no references, append fallback
            if answer.strip().endswith(":") and ("reference" not in answer.lower()):
                answer += "\nNo references could be extracted accurately from the document."
            # Optional: Add sources used section if reference titles are found in context
            sources_used = []
            for ref in cleaned_refs:
                if ref.get('title') and ref.get('title') in context:
                    sources_used.append(ref.get('title'))
            if sources_used:
                answer += "\n\n🔎 Sources Used:\n" + '\n'.join(f"{i+1}. {title}" for i, title in enumerate(sources_used))
            return answer.strip()
        except Exception as e:
            logging.error(f"Error during LLM generation: {str(e)}")
            return "Error generating response."


# Example usage
# if __name__ == "__main__":
#     runner = LLMRunner()
#     prompt = "What is the capital of France?"
#     context = ""
#     print(runner.generate(prompt, context)) 