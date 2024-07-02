from keras.models import Sequential
from keras.layers import Dense, LSTM


def build_model(num_features, num_units):
    model = Sequential([
        LSTM(units=num_units, input_shape=(None, num_features)),
        Dense(units=1, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
# model.summary()
