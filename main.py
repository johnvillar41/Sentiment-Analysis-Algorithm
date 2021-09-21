from TextCleaning import TextCleaning
from SentimentLogic import SentimentLogic
from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['GET'])
def welcome():
    return "<h3>Routes: /Vader/(Enter Sentence Message here for computation)</h3>"


@app.route('/Vader/<string:value>', methods=['GET'])
def Vader(value):
    return SentimentLogic.sentiment_scores_vader(value)


@app.route('/clean/<string:value>', methods=['GET'])
def tokenizeSample(value):
    textClean = TextCleaning(value)
    return textClean.pos_tagging()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
