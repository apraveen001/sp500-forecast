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
    (Version 2: Increased units, added dense layer)

    Args:
        input_shape: (window_size, num_features)
        lr: learning rate for Adam optimizer

    Returns:
        Compiled tf.keras Model
    """
    model = Sequential()

    # First LSTM block - Increased units
    # Input shape is specified here
    model.add(LSTM(150, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Second LSTM block - Increased units
    model.add(LSTM(75, return_sequences=False)) # return_sequences=False as it's the last LSTM layer
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Dense layers - Added one more dense layer
    model.add(Dense(128, activation='relu')) # Increased units
    model.add(Dropout(0.2)) # Optional dropout for dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) # Optional dropout for dense layers
    model.add(Dense(32, activation='relu'))

    # Regression output layer (predicting one value - the price)
    model.add(Dense(1, name='price')) # No activation for regression output

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error is common for regression
        metrics=['mae'] # Mean Absolute Error is easier to interpret
    )
    # Print model summary after building
    print("Built LSTM Model Summary:")
    model.summary()
    return model


def train_lstm(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200, # Increased epochs
    batch_size: int = 32,
    early_stopping_patience: int = 20 # Increased patience
):
    """
    Train the LSTM model with EarlyStopping and learning-rate reduction.
    (Version 2: Increased epochs and patience)

    Returns:
        History object from model.fit
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience, # Use parameter
            restore_best_weights=True,
            verbose=1 # Show message when stopping early
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=10, # Keep ReduceLROnPlateau patience moderate
            factor=0.5, # Reduce LR by half
            min_lr=1e-6, # Minimum learning rate
            verbose=1 # Show message when reducing LR
        )
    ]
    print(f"\n--- Starting LSTM Training ---")
    print(f"Max Epochs: {epochs}, Batch Size: {batch_size}, Early Stopping Patience: {early_stopping_patience}")
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1 # Show progress per epoch
    )
    print("--- Finished LSTM Training ---\n")
    return history


# Keep the test stub if you want to quickly test model building
if __name__ == '__main__':
    # quick test stub
    import numpy as np
    window_size, num_features = 20, 16 # Match your actual features
    X_dummy = np.random.rand(100, window_size, num_features)
    y_dummy = np.random.rand(100)
    model = build_lstm_model((window_size, num_features))
    # Dummy training call (won't run long)
    # hist = train_lstm(
    #     model,
    #     X_dummy[:80], y_dummy[:80],
    #     X_dummy[80:], y_dummy[80:],
    #     epochs=3 # Just run a few epochs for the stub test
    # )
    # print("Dummy training complete.")
    # print("Model summary printed above.")
    print("Model building test successful.")
