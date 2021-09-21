
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class TextCleaning:
   
    def __init__(self, textValue):
        self.textValue = textValue
        self.clean()        
        self.tokenize()

    def clean(self):
        text = re.sub('[^A-Za-z]+', ' ', self.textValue)
        return text

    def tokenize(self):
        tokens = nltk.word_tokenize(self.textValue)
        return tokens

    def pos_tagging(self):
        pos = nltk.pos_tag(self.tokenize())
        return str(pos)
