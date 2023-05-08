import re
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def break_into_chunks(text, length_of_chunks):
    # helper method that breaks string text into a list of sentences
    # each of whose elements are of length length_of_chunks
    # last words are thrown away if length_of_chunks does not divide len(text)
    list_of_words = text.split()
    if len(list_of_words)%length_of_chunks != 0:
        list_of_words = list_of_words[:len(list_of_words)-len(list_of_words)%length_of_chunks]
    list_of_words_np = np.asanyarray(list_of_words).reshape((int(len(list_of_words)/length_of_chunks), length_of_chunks))
    list_of_sentences = []
    for sentence_list in list_of_words_np:
        sentence = " ".join(sentence_list)
        list_of_sentences.append(sentence)
    return list_of_sentences

##### First we clean up the data and organize it s.t. every line has 15 words ######

file1 = open("data/Republic.txt", "r")
new_text = file1.read().replace('\n', ' ')
new_text = new_text.replace("--", " ")
new_text = re.sub(r'\[.*?\]', '', new_text)
new_text = re.sub(r'\*.*?\*', '', new_text)
new_text = re.sub(r'\{.*?\}', '', new_text)
new_text = re.sub(r'[^\w\s\'\"]', '', new_text)
new_text = new_text.replace("  ", " ")
new_text = new_text.replace("  ", " ")
new_text = new_text.replace("  ", " ")
list_of_sentences = break_into_chunks(new_text, 15)
file2 = open("data/NewRepublic.txt", 'w')
for sentence in list_of_sentences:
    file2.write(sentence + '\n')
file2.close()
# need to remove end of line, *blabla*, [blabla], {blabla}
file1.close()
print("The number of words in the Republic is: ")
print(len(new_text.split()))

file3 = open("data/Protagoras.txt", "r")
new_text = file3.read().replace('\n', ' ')
new_text = new_text.replace("--", " ")
new_text = new_text.replace("  ", " ")
new_text = new_text.replace("  ", " ")
new_text = new_text.replace("  ", " ")
new_text = re.sub(r'[^\w\s\'\"]', '', new_text)
list_of_sentences = break_into_chunks(new_text, 15)
file4 = open("data/NewProtagoras.txt", 'w')
for sentence in list_of_sentences:
    file4.write(sentence + '\n')
file4.close()
file3.close()



# Chapter and book markers manually removed from NE
file5 = open("data/NE.txt", 'r')
new_text = file5.read().replace('\n', ' ')
new_text = new_text.replace("--", " ")
new_text = new_text.replace("  ", " ")
new_text = new_text.replace("  ", " ")
new_text = new_text.replace("  ", " ")
new_text = re.sub(r'\[.*?\]', '', new_text)
new_text = re.sub(r'[^\w\s\'\"]', '', new_text)
list_of_sentences = break_into_chunks(new_text, 15)
file6 = open("data/NewNE.txt", 'w')
for sentence in list_of_sentences:
    file6.write(sentence + '\n')
file6.close()
file5.close()

list_of_sentences = ["this is the first sentence", "this is the second sentence"]
tensor = torch.tensor([sentence.split() for sentence in list_of_sentences])
print(tensor.shape)

"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased') 
indexed_tokens = tokenizer.encode(sentences[0], add_special_tokens=True)
segments_ids = [0 for _ in range(len(indexed_tokens))]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)
#print(encoded_layers.shape)
embedding = torch.mean(encoded_layers, dim=1).squeeze()

print(embedding)


tokens = tokenizer.tokenize(sentences[0])
tokens = ['[CLS]'] + tokens + ['[SEP]']
T = 20
padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
attn_mask=[1 if token != '[PAD]' else 0 for token in padded_tokens]
seg_ids=[0 for _ in range(len(padded_tokens))]
print("Segment Tokens are \n {}".format(seg_ids))
sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
print("senetence idexes \n {} ".format(sent_ids))
token_ids = torch.tensor(sent_ids).unsqueeze(0) 
attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
print(tokens)

hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
print(hidden_reps)
print(cls_head)
"""

"""
s_list = s.split()
s_lst_np = np.asanyarray(s_list)
print(s_lst_np)
numbers = np.zeros(100)


"""