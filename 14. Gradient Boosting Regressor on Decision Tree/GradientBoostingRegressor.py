import numpy as np
from DecisionTreeRegressor import *
import os
import json
from statistics import mean


class MyGradientBoostingRegressor():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param learning_rate, type:float
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        int (default=100)
        :param n_estimators, type: integer
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        :param max_depth, type: integer
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node

        estimators: the regression estimators
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree1 = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.estimators in this function
        '''
        F_mean = mean(y)

        F_i = [F_mean for ele in y]
        self.tree1 = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        self.tree1.fit(X,y)

        for i in range(self.n_estimators):   
            r_list = y-F_i
            r_im = np.array(r_list) # Into list
            
            self.estimators[i] = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.estimators[i].fit(X,r_im) # Fit to tree
            
            c_mj = [self.learning_rate*ele for ele in self.estimators[i].predict(X)]
            F_i = [sum(xi) for xi in zip(F_i, c_mj)]
        
        pass

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        # First mean value
        y0_mean = mean(self.tree1.predict(X))
        y_pred = [y0_mean for ele in self.tree1.predict(X)]
        
        # Sum of corrections
        for i in range(self.n_estimators):
            cor = self.estimators[i].predict(X)
            cor = [self.learning_rate*ele for ele in self.estimators[i].predict(X)]
            y_pred = [sum(xi) for xi in zip(y_pred, cor)]

        return y_pred

    def get_model_dict(self):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})

        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

def iterdict(x_row,d):
    if x_row[d['splitting_variable']]>d['splitting_threshold']:
        if isinstance(d['right'], dict):
            return iterdict(x_row,d['right'])
        else:            
            return d['right']

    else:
        if isinstance(d['left'], dict):
            return iterdict(x_row,d['left'])
        else:            
            return d['left']

# For test
if __name__=='__main__':
    for i in range(1):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            n_estimators = 10 + j * 10
            gbr = MyGradientBoostingRegressor(n_estimators=n_estimators, max_depth=5, min_samples_split=2)
            gbr.fit(x_train, y_train)
            model_dict = gbr.get_model_dict()

            y_pred = gbr.predict(x_train)

            with open("Test_data" + os.sep + "gradient_boosting_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_gradient_boosting_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")
