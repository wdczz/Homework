import numpy as np
class Evaluations():
    def __init__(self,pred,gt,classes):
        if type(pred)!=np.ndarray:
            pred=np.array(pred)
        if type(gt)!=np.array:
            gt=np.array(gt)
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0
        self.classes = classes
        for class_ in range(classes):
            index_ = class_
            tp_ = ((pred == index_)&(gt == index_)).sum()
            self.tp += tp_
            fn_ = ((pred != index_)&(gt == index_)).sum()
            self.fn += fn_
            fp_ = ((pred == index_)&(gt != index_)).sum()
            self.fp += fp_
            tn_ = ((pred != index_)&(gt != index_)).sum()
            self.tn += tn_

class Evaluation():
    def __init__(self, Evaluations):
        self.tp = Evaluations.tp
        self.fn = Evaluations.fn
        self.fp = Evaluations.fp
        self.tn = Evaluations.tn

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def accuracy(self):
        return (self.tp + self.tn) / (self.tn + self.tp + self.fn + self.fp)

    def f1_score(self):
        return 2 * self.tp / (2 * self.tp + self.fn + self.fp)