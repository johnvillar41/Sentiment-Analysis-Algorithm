from flask import jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from VaderModel import VaderModel

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
lemmatizer = WordNetLemmatizer()

class SentimentLogic:

    @staticmethod
    def sentiment_scores_vader(sentence):

        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(sentence)

        negativeVal = sentiment_dict['neg']*100
        neutralVal = sentiment_dict['neu']*100
        positiveVal = sentiment_dict['pos']*100
        compoundVal = sentiment_dict['compound']*100
        compoundScore = ""
        if sentiment_dict['compound'] >= 0.05:
            compoundScore = "Positive"

        elif sentiment_dict['compound'] <= - 0.05:
            compoundScore = "Negative"

        else:
            compoundScore = "Neutral"

        vaderModel = VaderModel(negativeVal, positiveVal,
                                neutralVal, compoundScore, compoundVal)
        return vaderModel.toJSON()

    @staticmethod
    def penn_to_wn(tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None

    @staticmethod
    def swn_polarity(text):
        """
        Return a sentiment polarity: 0 = negative, 1 = positive
        """
    
        sentiment = 0.0
        tokens_count = 0
    
        raw_sentences = sent_tokenize(text)
        for raw_sentence in raw_sentences:
            tagged_sentence = pos_tag(word_tokenize(raw_sentence))
    
            for word, tag in tagged_sentence:
                wn_tag = SentimentLogic.penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                    continue
    
                lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue
    
                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue
    
                # Take the first sense, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
    
                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
                tokens_count += 1
    
        # judgment call ? Default to positive or negative
        if not tokens_count:
            return 0
    
        # sum greater than 0 => positive sentiment
        if sentiment >= 0:
            return 1
    
        # negative sentiment
        return 0