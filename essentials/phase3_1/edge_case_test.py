from essentials.phase3_1.chunking import chunk_fixed, chunk_by_sentence, chunk_by_paragraph
from essentials.phase3_1.models import Chunk

# Edge case inputs
empty_text = ""
short_text = "Short text"
no_punctuation = "This is a paragraph with no punctuation and it just keeps going"

# Test edge cases
print('Testing edge cases...')

# Empty string
print('Empty String:')
print('Fixed:', len(chunk_fixed(empty_text, 50, 10)), 'chunks')
print('Sentence:', len(chunk_by_sentence(empty_text)), 'chunks')
print('Paragraph:', len(chunk_by_paragraph(empty_text)), 'chunks')

# Short text
print('\nShort Text:')
print('Fixed:', len(chunk_fixed(short_text, 50, 10)), 'chunks')
print('Sentence:', len(chunk_by_sentence(short_text)), 'chunks')
print('Paragraph:', len(chunk_by_paragraph(short_text)), 'chunks')

# No punctuation
print('\nNo Punctuation:')
print('Fixed:', len(chunk_fixed(no_punctuation, 50, 10)), 'chunks')
print('Sentence:', len(chunk_by_sentence(no_punctuation)), 'chunks')
print('Paragraph:', len(chunk_by_paragraph(no_punctuation)), 'chunks') 