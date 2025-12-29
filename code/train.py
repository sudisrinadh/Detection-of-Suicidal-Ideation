import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Embedding, Conv1D, MaxPooling1D, Dropout, Concatenate, BatchNormalization, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
import json

# Print TensorFlow and Keras versions for debugging
print("TensorFlow version:", tf.__version__)
try:
    import keras
    print("Keras version:", keras.__version__)
except:
    print("Keras version: Integrated with TensorFlow")

# Configuration
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
NUM_YESNO_FEATURES = 10
NUM_RATING_FEATURES = 10
NUM_TEXT_FEATURES = 8

# Define questions
QUESTIONS = {
    'yesno': [
        "Have you ever had thoughts about ending your life?",
        "Do you feel like you're a burden to others?",
        "Have you ever seriously considered suicide in the past year?",
        "Do you ever think people would be better off without you?",
        "Have you been isolating yourself from friends or family lately?",
        "Do you feel comfortable talking to someone about your emotional struggles?",
        "Have you ever spoken to a counselor or therapist?",
        "Have you stopped doing things you once enjoyed?",
        "Have you recently had trouble focusing or staying motivated?",
        "Do you experience trouble sleeping or eating due to emotional stress?"
    ],
    'rating': [
        "Rate how often you feel hopeless.",
        "Rate how often you feel emotionally exhausted.",
        "Rate your recent sleep quality.",
        "Rate your ability to enjoy daily activities.",
        "Rate your current level of stress.",
        "Rate your motivation to get out of bed in the morning.",
        "Rate how lonely you've felt in the past two weeks.",
        "Rate how often you feel overwhelmed by your responsibilities.",
        "Rate how safe you feel expressing your emotions to others.",
        "Rate how much support you feel from your friends/family."
    ],
    'text': [
        "What do you usually think about when you feel sad or overwhelmed?",
        "How have your emotions been over the past two weeks? Please describe.",
        "Do you often feel empty, hopeless, or disconnected from others? Explain if you'd like.",
        "If your emotions were a weather report today, what would it be and why?",
        "Finish this sentence: 'Lately, I've been feeling...'",
        "What do you usually do when you're feeling emotionally low?",
        "What kind of support would make you feel safer or more supported emotionally?",
        "What keeps you going on difficult days? What gives you hope?"
    ]
}

# Load Kaggle dataset
def load_kaggle_data(file_path= r"C:\Users\Administrator\OneDrive\Attachments\Desktop\cap\code\Suicide_Detection.csv"):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])
    df['class'] = df['class'].map({'suicide': 1, 'non-suicide': 0})
    return df['text'].values, df['class'].values

# Simulate responses to yes/no and rating questions
def simulate_structured_responses(texts, labels, num_yesno=10, num_rating=10):
    num_samples = len(texts)
    yesno_data = np.zeros((num_samples, num_yesno))
    rating_data = np.zeros((num_samples, num_rating))
    
    for i in range(num_samples):
        yesno_data[i] = np.random.binomial(1, 0.8 * labels[i] + 0.2 * (1 - labels[i]), num_yesno)
        rating_data[i] = np.random.randint(1, 11, num_rating) * (labels[i] * 0.8 + 0.2)
    
    return yesno_data, rating_data

# Convert binary labels to 1-10 scale
def binary_to_scale(labels, texts, tokenizer, max_len=100):
    scores = np.zeros(len(labels))
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    binary_model = tf.keras.Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    binary_model.fit(padded, labels, epochs=5, batch_size=64, verbose=0)
    probs = binary_model.predict(padded, verbose=0).flatten()
    scores = (probs * 9) + 1
    return scores

# Prepare text data
def prepare_text_data(texts, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences, tokenizer

# Build multi-input model
def build_model():
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
    embedding = Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM)(text_input)
    conv1 = Conv1D(128, 5, activation='relu', padding='same')(embedding)
    pool1 = MaxPooling1D(2)(conv1)
    conv2 = Conv1D(128, 5, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(2)(conv2)
    text_output = GlobalMaxPooling1D()(pool2)
    text_output = Dense(128, activation='relu')(text_output)
    text_output = Dropout(0.5)(text_output)
    
    yesno_input = Input(shape=(NUM_YESNO_FEATURES,), name='yesno_input')
    yesno_output = Dense(64, activation='relu')(yesno_input)
    yesno_output = BatchNormalization()(yesno_output)
    yesno_output = Dropout(0.3)(yesno_output)
    yesno_output = Dense(32, activation='relu')(yesno_output)
    
    rating_input = Input(shape=(NUM_RATING_FEATURES,), name='rating_input')
    rating_output = Dense(64, activation='relu')(rating_input)
    rating_output = BatchNormalization()(rating_output)
    rating_output = Dropout(0.3)(rating_output)
    rating_output = Dense(32, activation='relu')(rating_output)
    
    combined = Concatenate()([text_output, yesno_output, rating_output])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    output = Dense(1, activation='linear', name='output')(combined)
    
    model = Model(inputs=[text_input, yesno_input, rating_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png'):
    y_true_classes = np.round(y_true).astype(int)
    y_pred_classes = np.round(y_pred).astype(int)
    y_true_classes = np.clip(y_true_classes, 1, 10)
    y_pred_classes = np.clip(y_pred_classes, 1, 10)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 11), yticklabels=range(1, 11))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

# Main training function
def train_model():
    print("Loading Kaggle dataset...")
    texts, labels = load_kaggle_data(r"C:\Users\Administrator\OneDrive\Attachments\Desktop\cap\code\Suicide_Detection.csv")
    
    print("Simulating structured responses...")
    yesno_data, rating_data = simulate_structured_responses(texts, labels)
    
    print("Preprocessing text data...")
    text_sequences, tokenizer = prepare_text_data(texts)
    
    print("Converting labels to 1-10 scale...")
    target_scores = binary_to_scale(labels, texts, tokenizer, MAX_SEQUENCE_LENGTH)
    
    X_text_train, X_text_test, X_yesno_train, X_yesno_test, X_rating_train, X_rating_test, y_train, y_test = train_test_split(
        text_sequences, yesno_data, rating_data, target_scores, test_size=0.2, random_state=42
    )
    
    print("Building model...")
    model = build_model()
    model.summary()
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    print("Training model...")
    history = model.fit(
        [X_text_train, X_yesno_train, X_rating_train],
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Evaluating model...")
    test_loss, test_mae = model.evaluate(
        [X_text_test, X_yesno_test, X_rating_test],
        y_test,
        verbose=0
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    y_pred = model.predict([X_text_test, X_yesno_test, X_rating_test]).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")
    
    model.save('suicide_risk_model.h5')
    print("Model saved as suicide_risk_model.h5")
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved as tokenizer.pickle")
    
    with open('questions.json', 'w') as f:
        json.dump(QUESTIONS, f, indent=4)
    print("Questions saved as questions.json")
    
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    print("Plots saved: training_history.png, confusion_matrix.png")

if __name__ == "__main__":
    train_model()