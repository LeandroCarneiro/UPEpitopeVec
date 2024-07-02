from keras.models import Sequential
from keras.layers import Dense, LSTM


def build_model(length_array, num_features, num_units):
    model = Sequential([
        LSTM(units=num_units, input_shape=(length_array, num_features)),
        Dense(units=1, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


model = build_model(30, 17, 64)
model.summary()
