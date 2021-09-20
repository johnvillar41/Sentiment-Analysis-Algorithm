from flask import Flask
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
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
 
    print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        return("Positive")
 
    elif sentiment_dict['compound'] <= - 0.05 :
        return("Negative")
 
    else :
        return("Neutral")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
