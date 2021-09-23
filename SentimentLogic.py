from flask import jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from VaderModel import VaderModel

from nltk.corpus import sentiwordnet as swn

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
    def sentiwordnet(pos_data, lemma, synsets):
        sentiment = 0
        tokens_count = 0
        for word, pos in pos_data:
            if not pos:
                continue
                lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            if not lemma:
                continue
                synsets = wordnet.synsets(lemma, pos=pos)
            if not synsets:
                continue
                # Take the first sense, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
                tokens_count += 1
                # print(swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score())
            if not tokens_count:
                return 0
            if sentiment > 0:
                return "Positive"
            if sentiment == 0:
                return "Neutral"
            else:
                return "Negative"
