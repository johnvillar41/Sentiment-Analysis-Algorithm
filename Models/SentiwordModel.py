import json


class SentiwordModel:
    def __init__(self, polarityScore, positiveScore, negativeScore, sentimentScore, dict=None):
        self.polarityScore = polarityScore
        self.positiveScore = positiveScore
        self.negativeScore = negativeScore
        self.sentimentScore = sentimentScore
        if dict is not None:
            vars(self).update(dict)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
