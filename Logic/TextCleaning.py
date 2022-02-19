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

    @staticmethod
    def clean(text):
        text = re.sub('[^A-Za-z]+', ' ', text)
        return text

    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        return tokens

    @staticmethod
    def pos_tagging(tokens):
        pos = nltk.pos_tag(tokens)
        return pos

    @staticmethod
    def stop_words_remove(tokens, text):
        text = (" ").join(ele for ele in tokens if ele.lower()
                          not in stopwords.words('english'))
        return text

    @staticmethod
    def lemmatization(posTagged):
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

    @staticmethod
    def overallTextCleaning(text):
        _text = TextCleaning.clean(text)
        _tokens = TextCleaning.tokenize(_text)
        _text = TextCleaning.stop_words_remove(_tokens, _text)
        _pos = TextCleaning.pos_tagging(_tokens)
        _lemmas = TextCleaning.lemmatization(_pos)

        return _lemmas

    @staticmethod
    def checkIfWordExistOnWordNet(word):
        try:
            if word is None:
                return False
            if word.strip() == "":
                return False

            wn_lemmas = set(wordnet.all_lemma_names())
            if word in wn_lemmas:
                return True    
            else:
                return False
        except:
            return False

    @staticmethod
    def getSynonym(word):
        syn = list()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                syn.append(lemma.name()) 
        
        return syn
