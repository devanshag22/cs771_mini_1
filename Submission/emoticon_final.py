import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# Data Processing
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()

# Create a list of all emojis across the dataset
emojis = list(set([emoji for sample in train_emoticon_X for emoji in sample]))
encoder = OneHotEncoder(categories=[emojis]*13, sparse=False, handle_unknown='ignore')
emoji_sequences = [list(sample) for sample in train_emoticon_X]
encoded_X = encoder.fit_transform(emoji_sequences)

X_train = encoded_X
y_train = train_emoticon_Y

# Initialize the MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(128,),alpha=0.007171032073792162 ,max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Predict

test_df = pd.read_csv("datasets/test/test_emoticon.csv")
emoji_sequences_test = [list(sample) for sample in test_df['input_emoticon'].tolist()]
encoded_X_test = encoder.transform(emoji_sequences_test)
test_predictions = clf.predict(encoded_X_test)

# save the predictions to a txt file
np.savetxt("pred_emoticon.txt", test_predictions, fmt='%d')
print("Predictions saved to pred_emoticon.txt")

