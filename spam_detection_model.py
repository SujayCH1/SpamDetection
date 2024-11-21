import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Input
from sklearn.model_selection import train_test_split
import pickle
import os


MODEL_PATH = "spam_detector_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"


def load_and_preprocess_data():
    spam_data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    spam_data = spam_data.drop(columns=[col for col in spam_data.columns if "Unnamed" in col])
    spam_data.columns = ['label', 'data']
    spam_data['b_labels'] = spam_data['label'].map({'ham': 1, 'spam': 0})
    return spam_data

def train_and_save_model():
    data = load_and_preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(data['data'], data['b_labels'], test_size=0.33)
    
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(x_train)
    data_train = pad_sequences(tokenizer.texts_to_sequences(x_train))
    data_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=data_train.shape[1])
    

    D = 20  
    M = 15  
    T = data_train.shape[1]  
    V = len(tokenizer.word_index) 
    

    input_layer = Input(shape=(T,))
    embedding = Embedding(V + 1, D)(input_layer)
    lstm = LSTM(M, return_sequences=True)(embedding)
    pooling = GlobalMaxPooling1D()(lstm)
    output_layer = Dense(1, activation='sigmoid')(pooling)
    
    model = Model(input_layer, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(data_train, y_train, validation_data=(data_test, y_test), epochs=10, batch_size=32)
    
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    return model, tokenizer, T


def load_model_and_tokenizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        T = model.input_shape[1]
        return model, tokenizer, T
    else:
        return train_and_save_model()


model, tokenizer, T = load_model_and_tokenizer()


def predict_spam(email_text):
    email_seq = tokenizer.texts_to_sequences([email_text])
    email_pad = pad_sequences(email_seq, maxlen=T)
    prediction = model.predict(email_pad)[0][0]
    return "Spam" if prediction < 0.5 else "Ham"
