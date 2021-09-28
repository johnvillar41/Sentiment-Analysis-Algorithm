from flask import jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from VaderModel import VaderModel

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

