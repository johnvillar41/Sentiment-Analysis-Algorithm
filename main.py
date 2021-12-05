from Logic.TextCleaning import TextCleaning
from Logic.SentimentLogic import SentimentLogic
import Logic.SWNBigram as SWNBigram
from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['GET'])
def welcome():
    return "<h3>Routes: /Vader/(Enter Sentence Message here for computation)</h3>"

@app.route('/Vader/<string:value>', methods=['GET'])
def Vader(value):
    return SentimentLogic.applyVader(value).toJSON()

@app.route('/SentiWordNet/<string:value>', methods=['GET'])
def SWN(value):    
    return SentimentLogic.applySentiWordNet(value).toJSON()

@app.route('/Hybrid/<string:value>', methods=['GET'])
def Hybrid(value):    
    return SentimentLogic.applyHybrid(value).toJSON()

@app.route('/Clean/<string:value>', methods=['GET'])
def Clean(value):
    return str(TextCleaning.overallTextCleaning(value))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
