import re
import numpy as np

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
