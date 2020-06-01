import numpy as np
import os
import json
import operator
from statistics import mean

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.num_features = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        
        self.num_features = len(X[0])
        self.root = self.sub_tree(X, y, 0)

        return 0

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = []
        for i in range(len(X)):
            x_row =X[i,:]
            y_pred.append(iterdict(x_row,self.root))
        
        y_pred = [round(ele,6) for ele in y_pred]
        return y_pred

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

    def sub_tree(self, X, y, counter):
        
        if counter < self.max_depth:
            counter +=1
            
            if len(X)>= self.min_samples_split:
                xleft, xright, yleft, yright, var, thres = split_data(X, y, self.num_features)

                tree = {}
                tree['splitting_variable'] = var
                tree['splitting_threshold'] = thres

                if len(xleft) > 1:
                    tree['left'] = self.sub_tree(xleft ,yleft, counter)
                else:
                    tree['left'] = yleft[0]

                if len(xright) > 1:
                    tree['right'] = self.sub_tree(xright ,yright, counter)
                else:
                    tree['right'] = yright[0]

                return tree
            else:
                return mean(y)
        else:
            return mean(y)

def opt_split(X,y):  
    
    sse_min = 100000
    m_store=0
    n_store=0

    for m in range(len(X[0])):
        for n in range(len(X)):

            left = np.where(X[:,m] <= X[n,m])
            right = np.where(X[:,m] > X[n,m])

            if len(left[0])<len(X) and len(right[0])<len(X):
                yleft = y[left]
                yright = y[right]
                c1 = mean(yleft)
                c2 = mean(yright)

                sse1 = sum([(yi - c1)**2 for yi in yleft])
                sse2 = sum([(yi - c2)**2 for yi in yright])
                if sse_min > (sse1 + sse2):
                    sse_min = sse1 + sse2
                    m_store = m
                    n_store = n

    var = m_store
    thres = X[n_store,var]
    
    return var, thres

def split_data(X,y,n_features):
    # var: variable number
    if len(X)==2:
        var = 0
        
        if X[0,0]>X[1,0]:
            thres = X[1,0]
            left = [1]
            right = [0]
        else:
            thres = X[0,0]
            left = [0]
            right = [1]
    else:
        var, thres = opt_split(X,y)
        left = np.where(X[:,var] <= thres)
        right = np.where(X[:,var] > thres)
        
    xleft = np.squeeze(X[left,:])
    xright = np.squeeze(X[right,:])

    yleft = y[left]
    yright = y[right]

    if len(xleft)==n_features:
        xleft = np.expand_dims(xleft, axis=0)
    if len(xright)==n_features:
        xright = np.expand_dims(xright, axis=0)
    
    return xleft, xright, yleft, yright, var, thres

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

def compare_json_dic(json_dic, sample_json_dic):
    if isinstance(json_dic, dict):
        result = 1
        for key in sample_json_dic:
            if key in json_dic:
                result = result * compare_json_dic(json_dic[key], sample_json_dic[key])
                if result == 0:
                    return 0
            else:
                return 0
        return result
    else:
        rel_error = abs(json_dic - sample_json_dic) / np.maximum(1e-8, abs(sample_json_dic))
        if rel_error <= 1e-5:
            return 1
        else:
            return 0

def compare_predict_output(output, sample_output):
    rel_error = (abs(output - sample_output) / np.maximum(1e-8, abs(sample_output))).mean()
    if rel_error <= 1e-5:
        return 1
    else:
        return 0
    
# For test
if __name__=='__main__':
    for i in range(1):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_dict = tree.get_model_dict()
            y_pred = tree.predict(x_train)

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")



