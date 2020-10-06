from stanfordcorenlp import StanfordCoreNLP
import logging
import sys

local_corenlp_path = 'stanford-corenlp-4.0.0/stanford-corenlp-4.0.0'


nlp = StanfordCoreNLP(local_corenlp_path, quiet=False, logging_level=logging.DEBUG)

text = open('input.txt', 'r')
lines = text.readlines()
textt = ''
for i in lines:
  textt = textt + i

sys.stdout = open("output.txt", "w")

print('Tokenize:', nlp.word_tokenize(textt))
print('\n')
print('Part of Speech:', nlp.pos_tag(textt))
print('\n')
print('Named Entities:', nlp.ner(textt))
print('\n')
print('Constituency Parsing:', nlp.parse(textt))
print('\n')
print('Dependency Parsing:', nlp.dependency_parse(textt))
print('\n')


nlp.close()
sys.stdout.close()
