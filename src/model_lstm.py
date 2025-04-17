# src/model_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_model(
    input_shape: tuple,
    lr: float = 0.001
) -> tf.keras.Model:
    """
    Build and compile an LSTM model that predicts next-day price.

    Args:
        input_shape: (window_size, num_features)
        lr: learning rate for Adam optimizer

    Returns:
        Compiled tf.keras Model
    """
    model = Sequential()

    # First LSTM block
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Second LSTM block
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Regression output
    model.add(Dense(1, name='price'))

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model


def train_lstm(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32
):
    """
    Train the LSTM model with EarlyStopping and learning-rate reduction.

    Returns:
        History object from model.fit
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return history


if __name__ == '__main__':
    # quick test stub
    import numpy as np
    window_size, num_features = 20, 10
    X_dummy = np.random.rand(100, window_size, num_features)
    y_dummy = np.random.rand(100)
    model = build_lstm_model((window_size, num_features))
    hist = train_lstm(
        model,
        X_dummy[:80], y_dummy[:80],
        X_dummy[80:], y_dummy[80:]
    )
    print("Training complete. Final loss:", hist.history['loss'][-1])
    print("Validation loss:", hist.history['val_loss'][-1])
    print("Model summary:")
    model.summary()
    print("Model trained successfully.")
    
    # Save the model    
    model.save('../models/lstm_model.h5')
    print("Model saved as 'lstm_model.h5'.")