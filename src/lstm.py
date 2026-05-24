import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop


def load_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read().lower()


def prepare_data(text, max_length=40, step=3):
    vocabulary = sorted(list(set(text)))

    char_to_index = {char: index for index, char in enumerate(vocabulary)}
    index_to_char = {index: char for index, char in enumerate(vocabulary)}

    sentences = []
    next_chars = []

    for i in range(0, len(text) - max_length, step):
        sentences.append(text[i:i + max_length])
        next_chars.append(text[i + max_length])

    x = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=bool)
    y = np.zeros((len(sentences), len(vocabulary)), dtype=bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    return x, y, vocabulary, char_to_index, index_to_char


def build_model(max_length, vocabulary_size):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, vocabulary_size)))
    model.add(Dense(vocabulary_size))
    model.add(Activation("softmax"))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    return model


def sample_index(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)

    probabilities = np.random.multinomial(1, predictions, 1)

    return np.argmax(probabilities)


def generate_text(model, text, vocabulary, char_to_index, index_to_char,
                  max_length=40, length=6000, temperature=0.5):
    start_index = random.randint(0, len(text) - max_length - 1)

    sentence = text[start_index:start_index + max_length]
    generated = sentence

    for _ in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)), dtype=bool)

        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated


def main():
    input_path = os.path.join("src", "input.txt")
    output_dir = "result"
    output_path = os.path.join(output_dir, "gen.txt")

    os.makedirs(output_dir, exist_ok=True)

    text = load_text(input_path)

    max_length = 40
    step = 3

    x, y, vocabulary, char_to_index, index_to_char = prepare_data(
        text,
        max_length=max_length,
        step=step
    )

    model = build_model(max_length, len(vocabulary))

    print("Model summary:")
    model.summary()

    print("Training started...")
    model.fit(x, y, batch_size=128, epochs=20)

    print("Generating text...")
    generated = generate_text(
        model,
        text,
        vocabulary,
        char_to_index,
        index_to_char,
        max_length=max_length,
        length=8000,
        temperature=0.5
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(generated)

    print("Generated text saved to result/gen.txt")


if __name__ == "__main__":
    main()
