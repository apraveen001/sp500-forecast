# src/model_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_model(
    input_shape: tuple,
    lr: float = 0.0005
) -> tf.keras.Model:
    """
    Build and compile an LSTM model that predicts next-day price.
    (Version 3: Simpler architecture, similar to original)

    Args:
        input_shape: (window_size, num_features)
        lr: learning rate for Adam optimizer

    Returns:
        Compiled tf.keras Model
    """
    model = Sequential()

    # First LSTM block
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.25)) # Adjusted dropout slightly
    model.add(BatchNormalization())

    # Second LSTM block
    model.add(LSTM(64, return_sequences=False)) # Last LSTM layer
    model.add(Dropout(0.25)) # Adjusted dropout slightly
    model.add(BatchNormalization())

    # Dense layers
    model.add(Dense(64, activation='relu'))
    # Removed extra dense layers from previous attempt
    model.add(Dense(32, activation='relu'))

    # Regression output layer
    model.add(Dense(1, name='price'))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    print("Built LSTM Model Summary (Simpler Architecture):")
    model.summary()
    return model


def train_lstm(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 150, # Moderate epochs
    batch_size: int = 32,
    early_stopping_patience: int = 15 # Moderate patience
):
    """
    Train the LSTM model with EarlyStopping and learning-rate reduction.
    (Version 3: Moderate epochs and patience)

    Returns:
        History object from model.fit
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=7, # Reduce LR patience slightly less than stopping patience
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    print(f"\n--- Starting LSTM Training (Simpler Model) ---")
    print(f"Max Epochs: {epochs}, Batch Size: {batch_size}, Early Stopping Patience: {early_stopping_patience}")
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print("--- Finished LSTM Training ---\n")
    return history

# Keep the test stub if needed
if __name__ == '__main__':
    import numpy as np
    window_size, num_features = 20, 16 # Match your actual features
    model = build_lstm_model((window_size, num_features))
    print("Model building test successful.")

