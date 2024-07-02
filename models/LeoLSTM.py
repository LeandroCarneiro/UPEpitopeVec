from keras.models import Sequential
from keras.layers import Dense, LSTM


def build_model(length_array, num_features, num_units):
    model = Sequential([
        LSTM(units=num_units, input_shape=(length_array, num_features)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# model = build_model(30, 17, 64)
# model.summary()
