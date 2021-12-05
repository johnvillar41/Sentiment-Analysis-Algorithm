import json
class HybridModel:
    def __init__(self, hybridScore, hybridValue):       
        self.hybridScore = hybridScore
        self.hybridValue = hybridValue

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)