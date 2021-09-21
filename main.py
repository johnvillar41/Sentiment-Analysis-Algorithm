from SentimentLogic import SentimentLogic
from flask import Flask

app = Flask(__name__)


@app.route('/Welcome/<int:value>', methods=['GET', 'POST'])
def welcome(value):
    return "Hello World!" + str(value+10)


@app.route('/Vader/<string:value>', methods=['GET', 'POST'])
def Vader(value):
    return SentimentLogic.sentiment_scores_vader(value)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
