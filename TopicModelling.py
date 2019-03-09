import csv
import os
import pickle
import gensim
import pyLDAvis.gensim

from stop_words import get_stop_words
stop_wordsNL = get_stop_words('nl')
stop_wordsEN = get_stop_words('en')
stop_wordsManual = ['Geert_Wilders', 'shared', 'link', 'And', 'The', 'will', '!!', 'willen', 'zullen', 'But', 'moeten',
                    'waar', 'wij', 'gaan', 'komen', '...', '..', 'mee', 'houden', 'weer', 'staan', 'maken', 'mit',
                    'und', 'wel', 'zeggen', 'ander', ';-)', '.....', 'geel', 'hes']

from gensim import corpora


#replace 'DirectoryWithTheFiles' with the filepath of directory with the textfiles
def get_word(filepath):
    with open(r'DirectoryWithTheFiles' + "\\" + filepath,
              encoding="utf8") as tsvfile:
        reader = csv.DictReader(tsvfile,
                                fieldnames=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8',
                                            'col9', 'col10'], dialect='excel-tab')

#filter Dutch, English and party-specific stop words from the text post
        data = []
        for row in reader:
            for header, value in row.items():
                if header == 'col3' and len(value) > 1 and value not in stop_wordsNL and value not in stop_wordsEN:
                    if value not in stop_wordsManual:
                        data.append(value)
    return data


#replace 'DirectoryWithTheFiles' with the filepath of the directory with the text files for topic modelling
directory = os.fsencode(r'DirectoryWithTheFiles')

#iterate through all textposts in the specified directory, to filter them one by one and to tokenize them
text_data = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    k = 0
    #change accordingly if textfiles have a different extension (in this case it was .out)
    if filename.endswith(".out"):
        tokens = get_word(filename)
        if len(tokens) > 0:
            text_data.append(tokens)
            k += 1

#build a corpus from the dictionary
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

#use lda to model the topics, specified at 5 topics
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

#prepare results for visualization
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

#visualize the LDA topic modelling results using pyLDAvis
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display, 'modellingGH.html')
pyLDAvis.show(lda_display)


#other options for topic modelling, for example 3 topics (narrower) or 10 topics (broader):

# NUM_TOPICS = 3
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('model3.gensim')
#topics = ldamodel.print_topics(num_words=4)
#for topic in topics:
#    print(topic)

#NUM_TOPICS = 10
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('model10.gensim')
#topics = ldamodel.print_topics(num_words=4)
#for topic in topics:
#    print(topic)

#lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
#lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
#pyLDAvis.save_html(lda_display, 'modellingPVV10topics.html')
#pyLDAvis.show(lda_display10)