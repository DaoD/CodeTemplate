import numpy as np
np.random.seed(0)


class Metrics(object):
    def __init__(self, score_file_path:str):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path

    def __read_socre_file(self, score_file_path):
        pass

    def evaluate_all_metrics(self):
        pass
