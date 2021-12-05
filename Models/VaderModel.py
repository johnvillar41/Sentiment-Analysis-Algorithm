import json


class VaderModel:
    def __init__(self, negativeValue, positiveValue, neutralValue, compoundScore, compoundValue, dict=None):
        self.negativeValue = negativeValue
        self.positiveValue = positiveValue
        self.neutralValue = neutralValue
        self.compoundScore = compoundScore
        self.compoundValue = compoundValue
        if dict is not None:
            vars(self).update(dict)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
