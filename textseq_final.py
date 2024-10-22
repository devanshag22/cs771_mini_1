import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, GRU


# read text sequence dataset

test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")

test_X = test_seq_X['input_str'].apply(lambda x: pd.Series(list(x)))
test_X.columns = [f'col_{i+1}' for i in range(50)]
test_X = test_X.astype(int)

# load the model
model = models.load_model('models/gru.h5')
# predict
predictions = model.predict(test_X)
# save predictions in txt file
np.savetxt('pred_textseq.txt', predictions, fmt='%d')