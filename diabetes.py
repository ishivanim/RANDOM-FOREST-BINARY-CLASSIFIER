import streamlit as st
import joblib
import numpy as np
from collections import Counter

# Load trained model
class Node():
    def __init__(self, features = None, threshold = None, left = None, right = None,*, value = None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def _is_leaf_node(self):
        return self.value is not None
    
class _Decision_Tree_():
    def __init__(self, min_sample_split = 3, max_depth = 200, n_features = None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.roots = None

    def fit(self, x,y):
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._grow_tree_(x,y)
    
    def _grow_tree_(self,x,y,depth = 0):
        n_samples, n_feature = x.shape
        n_labels = len(np.unique(y))

        # stopping condition
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            leaf_value = self.most_common_value(y)
            return Node(value = leaf_value)
        
        feats = np.random.choice(n_feature, self.n_features, replace=False)
        best_features, best_threshold = self._best_split_(x,y,feats)

        left_idxs, right_idxs = self._split_(x[:, best_features], best_threshold)
        left = self._grow_tree_(x[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree_(x[right_idxs,:], y[right_idxs], depth+1)

        return Node(best_features, best_threshold, left, right)

    def most_common_value(self,y):
        #y=np.ravel(y)
        counter = Counter(y)
        common_value = counter.most_common(1)[0][0]
        return common_value
    
    def _best_split_(self,x,y, n_feats):
        best_gain = -1
        split_idxs, split_threshold = None, None

        for idxs in n_feats:
            x_column = x[:,idxs]
            threshold = np.unique(x_column)

            for thr in threshold:
                gain = self._information_gain_(x_column,y,thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idxs = idxs
                    split_threshold = thr
            
        return split_idxs, split_threshold

    def _information_gain_(self,x,y,thrs):
        parent_entropy = self._entropy_(y)

        left_idxs, right_idxs = self._split_(x,thrs)

        if len(left_idxs) == 0 or len(right_idxs) == 0 :
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy_(y[left_idxs]), self._entropy_(y[right_idxs])

        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        info = parent_entropy - child_entropy
        return info

    def _entropy_(self,y):
        #y = y.ravel()
        hist = np.bincount(y)
        ps = hist/len(y)
        ent = -np.sum([p*np.log(p) for p in ps if p>0])
        return ent
    
    def _split_(self, x,thrsh):
        left = np.argwhere(x <= thrsh).flatten()
        right = np.argwhere(x > thrsh).flatten()
        return left, right
    
    def predict(self,x):
        pred = self._traverse_tree_(x, self.root)
        return pred
    
    def _traverse_tree_(self, x, node):
        if node._is_leaf_node(): 
            return [1.0 if node.value == 0 else 0.0, 1.0 if node.value == 1 else 0.0]
        
        if x[node.features] <= node.threshold:
            return self._traverse_tree_(x,node.left)
        return self._traverse_tree_(x, node.right)

class Random_forest():
    def __init__(self, min_sample_split = 2, max_depth = 10, n_trees = 20, n_features = None):
        self.n_features = n_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_trees =n_trees
        self.trees = []

    def fit(self,x,y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = _Decision_Tree_(min_sample_split = self.min_sample_split, max_depth = self.max_depth, n_features = self.n_features)

            x_sample, y_sample = self.bootstrap(x,y)
            tree.fit(x_sample,y_sample)
            self.trees.append(tree)

    def bootstrap(self,x,y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace= True)
        return x[idxs], y[idxs]

    def predict(self,x):
        tree_probs = np.array([tree.predict(xi) for tree in self.trees for xi in x])
        tree_probs = tree_probs.reshape(len(self.trees), len(x),2)
        avg_predictions = np.mean(tree_probs, axis=0)
        return avg_predictions

    def most_common_value(self,y):
        counter = Counter(y)
        commo = counter.most_common(1)[0][0]
        return commo
    
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below to predict if they are likely to have diabetes.")

# Inputs (adjust to match your dataset)
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    # Prepare input as 2D array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    probs = model.predict(input_data)[0] 
    prediction = int(np.argmax(probs))    
    confidence = probs[prediction]


    if prediction == 1:
        st.error(f"🧪 Prediction: You are Diabetic (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ Prediction: You are not Diabetic (Confidence: {confidence:.2f})")
