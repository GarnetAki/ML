import numpy as np
from tqdm import tqdm
from KDTree import MyGeniousKDTree

class MyGeniousKNN:

    def __init__(self, k) -> None:
        self.k = k


    def fit(self, x, y):
        if len(x) != len(y):
            raise Exception("len x != len y")
        x['Win'] = y
        self.node = MyGeniousKDTree.build_kd_tree(x.values.tolist())
        return self


    def prediction(self, neighbors):
        count = {}
        for instance in neighbors:
            if instance[-1] in count:
                count[instance[-1]] +=1
            else :
                count[instance[-1]] = 1

        target = max(count.items(), key=lambda x: x[1])[0]
        return target


    def get_neighbors(self, test):
        return MyGeniousKDTree.nearest_neighbors(self.node, test, self.k)


    def predict(self, test_x):
        test_predictions = np.zeros(len(test_x))
        for i in tqdm(range(len(test_x))):
            test_predictions[i] = self.prediction(self.get_neighbors(test_x.iloc[i]))

        return test_predictions


    def accuracy(self, test_y, test_prediction):
        correct = 0
        for i in range (len(test_y)):
            if test_y.iloc[i] == test_prediction[i]:
                correct += 1

        return (correct / len(test_y))