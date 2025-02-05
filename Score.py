class Score:
    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __str__(self):
        return f'Accuracy={self.accuracy}\t Precision={self.precision}\t recall={self.recall}\tF1={self.f1}'
