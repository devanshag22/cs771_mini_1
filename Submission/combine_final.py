
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Data Reading and processing

train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_seq_X = train_seq_df['input_str']
train_seq_Y = train_seq_df['label']
valid_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
valid_seq_X = valid_seq_df['input_str']
valid_seq_Y = valid_seq_df['label']

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon']
train_emoticon_Y = train_emoticon_df['label']
valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon']
valid_emoticon_Y = valid_emoticon_df['label']

train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

train_feat_X = np.array([x.flatten() for x in train_feat_X])
valid_feat_X = np.array([x.flatten() for x in valid_feat_X])

train_seq_X = train_seq_df['input_str'].apply(lambda x: pd.Series(list(x)))
train_seq_X.columns = [f'col_{i+1}' for i in range(50)]
train_seq_X = train_seq_X.astype(int)
valid_seq_X = valid_seq_df['input_str'].apply(lambda x: pd.Series(list(x)))
valid_seq_X.columns = [f'col_{i+1}' for i in range(50)]
valid_seq_X = valid_seq_X.astype(int)


emojis = list(set([emoji for sample in train_emoticon_X for emoji in sample]))
encoder = OneHotEncoder(categories=[emojis]*13, sparse=False, handle_unknown='ignore')

emoji_sequences = [list(sample) for sample in train_emoticon_X]
emoji_sequences_val = [list(sample) for sample in valid_emoticon_X]
train_emoticon_X = encoder.fit_transform(emoji_sequences)
valid_emoticon_X = encoder.transform(emoji_sequences_val)

train_X = np.concatenate([train_seq_X, train_emoticon_X, train_feat_X], axis=1)
valid_X = np.concatenate([valid_seq_X, valid_emoticon_X, valid_feat_X], axis=1)

# Model Training
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, seed=42)
model.fit(train_X, train_seq_Y)

# Predicting on test data

test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")
test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")
test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)
test_feat_X = np.array([x.flatten() for x in test_feat_X['features']])

emoji_sequences_test = [list(sample) for sample in test_emoticon_X['input_emoticon']]
test_emoticon_X = encoder.transform(emoji_sequences_test)

test_seq_X = test_seq_X['input_str'].apply(lambda x: pd.Series(list(x)))
test_seq_X.columns = [f'col_{i+1}' for i in range(50)]
test_seq_X = test_seq_X.astype(int)

test_X = np.concatenate([test_seq_X, test_emoticon_X, test_feat_X], axis=1)

preds_test = model.predict(test_X)
# save preds_test to txt file
np.savetxt("pred_combined.txt", preds_test, fmt="%d")
print("Predictions saved to pred_combined.txt")