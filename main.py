from AppKey import require_appkey
from Logic.TextCleaning import TextCleaning
from Logic.SentimentLogic import SentimentLogic
from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['GET'])
@require_appkey
def welcome():
    return "<h3>Routes: /Vader/(Enter Sentence Message here for computation)</h3>"

@app.route('/Vader/<string:value>', methods=['GET'])
@require_appkey
def Vader(value):
    return SentimentLogic.applyVader(value).toJSON()

@app.route('/SentiWordNet/<string:value>', methods=['GET'])
@require_appkey
def SWN(value):    
    return SentimentLogic.applySentiWordNet(value).toJSON()

@app.route('/Hybrid/<string:value>', methods=['GET'])
@require_appkey
def Hybrid(value):    
    return SentimentLogic.applyHybrid(value).toJSON()

@app.route('/Clean/<string:value>', methods=['GET'])
@require_appkey
def Clean(value):
    return str(TextCleaning.overallTextCleaning(value))

@app.route('/Check/<string:value>', methods=['GET'])
@require_appkey
def Check(value):
    return str(TextCleaning.checkIfWordExistOnWordNet(value)).lower()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
