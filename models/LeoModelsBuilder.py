from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU


def build_LSTM_model(length_array, num_features, num_units):
    model = Sequential([
        LSTM(units=num_units, input_shape=(length_array, num_features), dropout=0.2, recurrent_dropout=0, activation='tanh',
             recurrent_activation='sigmoid', unroll=True, use_bias=True),
        # LSTM(units=196, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def build_RNN_model(length_array, num_features, num_units):
    model = Sequential([
        SimpleRNN(units=num_units, input_shape=(length_array, num_features)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def build_GRU_model(length_array, num_features, num_units):
    model = Sequential([
        GRU(units=num_units, input_shape=(length_array, num_features)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def build_MLP_model(length_array, num_features, num_units):
    model = Sequential([
        # Input layer
        Dense(num_units, activation='relu',
              input_shape=(length_array, num_features)),

        # Hidden layers
        Dense(int(num_units*2), activation='relu'),
        Dense(int(num_units*2), activation='relu'),

        # Output layer
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_MLP_embedder(embed_size, size_vocabulary):
    model = Sequential([
        Dense(embed_size, activation='linear'),
        Dense(size_vocabulary, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
# model = build_model(30, 17, 64)
# model.summary()
