import json


class SentiwordModel:
    def __init__(self, polarityScore, positiveScore, negativeScore, sentimentScore):
        self.polarityScore = polarityScore      
        self.positiveScore = positiveScore
        self.negativeScore = negativeScore       
        self.sentimentScore = sentimentScore

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
