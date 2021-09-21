import json


class VaderModel:
    def __init__(self, negativeValue, positiveValue, neutralValue, compoundScore, compoundValue):
        self.negativeValue = negativeValue
        self.positiveValue = positiveValue
        self.neutralValue = neutralValue
        self.compoundScore = compoundScore
        self.compoundValue = compoundValue

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
