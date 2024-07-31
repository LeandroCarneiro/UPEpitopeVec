from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, SpatialDropout1D, Dropout
from keras.regularizers import l2
from keras.constraints import max_norm


def build_LSTM_model(length_array, num_features, num_units):
    model = Sequential([
        SpatialDropout1D(0.2),
        LSTM(units=num_units, input_shape=(length_array, num_features), dropout=0.2, recurrent_dropout=0, activation='tanh',
             recurrent_activation='sigmoid', unroll=True, use_bias=True),
        Dropout(0.5),
        # LSTM(units=196, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# units: The dimensionality of the output space. This defines how many neurons the RNN layer will have.
# activation: The activation function to use. Common choices are 'tanh', 'relu', and 'sigmoid'.
# use_bias: Whether the layer uses a bias vector. It's usually beneficial to keep this enabled.
# kernel_initializer, recurrent_initializer, bias_initializer: Initializers determine how the weights are set before training starts. Different initializers can lead to different training dynamics.
# kernel_regularizer, recurrent_regularizer, bias_regularizer: Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. These are useful for preventing overfitting.
# kernel_constraint, recurrent_constraint, bias_constraint: Constraints allow you to apply constraints on layer parameters during optimization.
# dropout, recurrent_dropout: Dropout rates for the linear transformation of inputs and recurrent state. These help prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
# return_sequences: Whether to return the last output in the output sequence, or the full sequence. Useful for stacking RNN layers.
# return_state: Whether to return the last state in addition to the output. Useful for applications where you need the final state.
# go_backwards: If True, the input sequence is processed backward. This can sometimes improve performance.
# stateful: If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch. This is useful for stateful RNNs.
# unroll: If True, the network will be unrolled, which can speed up training but increase memory usage. Suitable for short sequences.
# seed: Random number generator seed for reproducibility.
def build_RNN_model(length_array, num_features, num_units):
    model = Sequential([
        SimpleRNN(
            units=num_units,  # Dimensionality of the output space
            input_shape=(length_array, num_features),
            activation="relu",  # Activation function to use; tanh is the default
            use_bias=True,  # Whether the layer uses a bias vector; True is the default
            # Initializer for the kernel weights matrix
            kernel_initializer="glorot_uniform",
            # Initializer for the recurrent_kernel weights matrix
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",  # Initializer for the bias vector
            # Regularizer function for the kernel weights matrix
            kernel_regularizer=l2(0.01),
            # Regularizer function for the recurrent_kernel weights matrix
            recurrent_regularizer=l2(0.01),
            # Regularizer function for the bias vector
            bias_regularizer=l2(0.01),
            activity_regularizer=None,  # Regularizer function for the output of the layer
            # Constraint function for the kernel weights matrix
            kernel_constraint=max_norm(3.),
            # Constraint function for the recurrent_kernel weights matrix
            recurrent_constraint=max_norm(3.),
            bias_constraint=None,  # Constraint function for the bias vector
            dropout=0.25,  # Fraction of the units to drop for the linear transformation of the inputs
            # Fraction of the units to drop for the linear transformation of the recurrent state
            recurrent_dropout=0.25,
            return_state=False,  # Whether to return the last state in addition to the output
            go_backwards=False,  # If True, process the input sequence backwards
            stateful=False,  # If True, the last state for each sample will be used as initial state for the next batch
            unroll=False,  # If True, the network will be unrolled; suitable for short sequences
            seed=None,  # Random number generator seed
            return_sequences=False,  # Whether to return the last output or the full sequence
        ),

        # SimpleRNN(units=num_units, input_shape=(length_array, num_features)),
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
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_MLP_embedder(embed_size, size_vocabulary):
    model = Sequential([
        Dense(embed_size, activation='relu'),
        Dense(size_vocabulary, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
# model = build_model(30, 17, 64)
# model.summary()
