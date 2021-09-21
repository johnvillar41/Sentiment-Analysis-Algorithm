
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB,
            'N': wordnet.NOUN, 'R': wordnet.ADV}
wordnet_lemmatizer = WordNetLemmatizer()


class TextCleaning:

    def __init__(self, textValue):
        self.textValue = textValue

    def clean(self):
        self.textValue = re.sub('[^A-Za-z]+', ' ', self.textValue)

    def tokenize(self):
        tokens = nltk.word_tokenize(self.textValue)
        return tokens

    def pos_tagging(self):
        pos = nltk.pos_tag(self.tokenize())
        return pos

    def stop_words_remove(self, tokens):
        self.new_text = (" ").join(ele for ele in tokens if ele.lower()
                                   not in stopwords.words('english'))

    def lemmatization(self, posTagged):
        lemma = []
        pos = posTagged
        for ele, tag in pos:
            tag = pos_dict.get(tag[0])
            if ele.lower() not in stopwords.words('english'):
                if not tag:
                    lemma.append(ele)
                else:
                    lemma.append(wordnet_lemmatizer.lemmatize(ele, tag))
        return lemma

    def finalTextForm(self):
        self.clean()
        tokens = self.tokenize()
        self.stop_words_remove(tokens)
        posTags = self.pos_tagging()
        lemmas = self.lemmatization(posTags)
        return str(lemmas)
       