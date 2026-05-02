from essentials.phase3_1.chunking import chunk_fixed, chunk_by_sentence, chunk_by_paragraph
from essentials.phase3_1.models import Chunk

# Read the example input text
with open('essentials/phase3_1/example_input.txt', 'r') as file:
    text = file.read()

# Run chunking strategies
fixed_chunks = chunk_fixed(text, 50, 10)
sentence_chunks = chunk_by_sentence(text)
paragraph_chunks = chunk_by_paragraph(text)

# Print results
print('Fixed:', len(fixed_chunks), 'chunks')
print('Sentence:', len(sentence_chunks), 'chunks')
print('Paragraph:', len(paragraph_chunks), 'chunks')

print('First Fixed Chunk:', fixed_chunks[0].text)
print('First Sentence Chunk:', sentence_chunks[0].text)
print('First Paragraph Chunk:', paragraph_chunks[0].text) 