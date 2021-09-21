from VaderModel import VaderModel
from flask import Flask
from flask import jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = Flask(__name__)


@app.route('/Welcome/<int:value>', methods=['GET', 'POST'])
def welcome(value):
    return "Hello World!" + str(value+10)


@app.route('/Vader/<string:value>', methods=['GET', 'POST'])
def Vader(value):
    return sentiment_scores(value)


def sentiment_scores(sentence):

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
