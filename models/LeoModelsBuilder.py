from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU


def build_LSTM_model(length_array, num_features, num_units):
    model = Sequential([
        LSTM(units=num_units, input_shape=(length_array, num_features)),
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


# model = build_model(30, 17, 64)
# model.summary()
