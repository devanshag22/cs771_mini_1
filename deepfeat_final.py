import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Data Reading

train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
val_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
val_feat_X = val_feat['features']
val_feat_Y = val_feat['label']

# flatten the features  
train_feat_X = np.array([x.flatten() for x in train_feat_X])
val_feat_X = np.array([x.flatten() for x in val_feat_X])

clf = RandomForestClassifier(n_estimators=250, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=10, random_state=2)
clf.fit(train_feat_X, train_feat_Y)


# Prediction
test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)
test_feat_X = np.array([x.flatten() for x in test_feat_X['features']])
y = clf.predict(test_feat_X)

np.savetxt("pred_deepfeat.txt", y, fmt='%d')
print("Predictions saved to pred_deepfeat.txt")

