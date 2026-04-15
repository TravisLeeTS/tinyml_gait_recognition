from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


REPRODUCTION_MODEL_NAMES = [
    "ConvLSTM",
    "CNN-GRU",
    "CNN-BiGRU",
    "CNN-BiLSTM",
    "CNN-LSTM",
]


def _cnn_rnn(
    *,
    name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    rnn_layer: type[layers.Layer],
    bidirectional: bool,
    conv_filters: tuple[int, int],
    subsequences: int,
    hidden_size: int = 100,
    dense_size: int = 100,
    dropout: float = 0.5,
) -> keras.Model:
    timesteps, channels = input_shape
    if timesteps % subsequences != 0:
        raise ValueError(f"Timesteps {timesteps} must divide into {subsequences} subsequences")
    subseq_len = timesteps // subsequences

    inputs = keras.Input(shape=input_shape, name="imu_window")
    x = layers.Reshape((subsequences, subseq_len, channels), name="subsequence_view")(inputs)
    x = layers.TimeDistributed(
        layers.Conv1D(conv_filters[0], kernel_size=3, padding="same", activation="relu"),
        name="td_conv_1",
    )(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name="td_dropout_1")(x)
    x = layers.TimeDistributed(
        layers.Conv1D(conv_filters[1], kernel_size=3, padding="same", activation="relu"),
        name="td_conv_2",
    )(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name="td_dropout_2")(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2), name="td_max_pool")(x)
    x = layers.TimeDistributed(layers.Flatten(), name="td_flatten")(x)

    rnn = rnn_layer(hidden_size, name=name.lower().replace("-", "_") + "_rnn")
    if bidirectional:
        x = layers.Bidirectional(rnn, name="bidirectional_rnn")(x)
    else:
        x = rnn(x)
    x = layers.Dropout(dropout, name="rnn_dropout")(x)
    x = layers.Dense(dense_size, activation="relu", name="dense_100")(x)
    x = layers.Dropout(dropout, name="dense_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity")(x)
    return keras.Model(inputs, outputs, name=name.lower().replace("-", "_"))


def _conv_lstm(
    *,
    input_shape: tuple[int, int],
    num_classes: int,
    subsequences: int,
    dropout: float = 0.5,
) -> keras.Model:
    timesteps, channels = input_shape
    if timesteps % subsequences != 0:
        raise ValueError(f"Timesteps {timesteps} must divide into {subsequences} subsequences")
    subseq_len = timesteps // subsequences

    inputs = keras.Input(shape=input_shape, name="imu_window")
    x = layers.Reshape((subsequences, subseq_len, channels), name="subsequence_view")(inputs)
    x = layers.ConvLSTM1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        dropout=dropout,
        name="conv_lstm_1d",
    )(x)
    x = layers.Dropout(dropout, name="conv_lstm_dropout")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(100, activation="relu", name="dense_100")(x)
    x = layers.Dropout(dropout, name="dense_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity")(x)
    return keras.Model(inputs, outputs, name="convlstm")


def build_reproduction_model(
    name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    subsequences: int = 4,
) -> keras.Model:
    normalized = name.lower().replace("_", "-")
    if normalized == "convlstm":
        return _conv_lstm(input_shape=input_shape, num_classes=num_classes, subsequences=subsequences)
    if normalized == "cnn-gru":
        return _cnn_rnn(
            name="CNN-GRU",
            input_shape=input_shape,
            num_classes=num_classes,
            rnn_layer=layers.GRU,
            bidirectional=False,
            conv_filters=(32, 128),
            subsequences=subsequences,
        )
    if normalized == "cnn-bigru":
        return _cnn_rnn(
            name="CNN-BiGRU",
            input_shape=input_shape,
            num_classes=num_classes,
            rnn_layer=layers.GRU,
            bidirectional=True,
            conv_filters=(32, 128),
            subsequences=subsequences,
        )
    if normalized == "cnn-bilstm":
        return _cnn_rnn(
            name="CNN-BiLSTM",
            input_shape=input_shape,
            num_classes=num_classes,
            rnn_layer=layers.LSTM,
            bidirectional=True,
            conv_filters=(64, 64),
            subsequences=subsequences,
        )
    if normalized == "cnn-lstm":
        return _cnn_rnn(
            name="CNN-LSTM",
            input_shape=input_shape,
            num_classes=num_classes,
            rnn_layer=layers.LSTM,
            bidirectional=False,
            conv_filters=(64, 64),
            subsequences=subsequences,
        )
    raise ValueError(f"Unknown reproduction model: {name}")
