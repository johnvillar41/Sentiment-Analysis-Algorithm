from TextCleaning import TextCleaning
from SentimentLogic import SentimentLogic
import SWNBigram
from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['GET'])
def welcome():
    return "<h3>Routes: /Vader/(Enter Sentence Message here for computation)</h3>"

@app.route('/Vader/<string:value>', methods=['GET'])
def Vader(value):
    return SentimentLogic.applyVader(value)

@app.route('/SentiWordNet/<string:value>', methods=['GET'])
def SWN(value):    
    return SentimentLogic.applySentiWordNet(value)

@app.route('/Clean/<string:value>', methods=['GET'])
def Clean(value):
    textClean = TextCleaning(value)
    return textClean.finalTextForm()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
