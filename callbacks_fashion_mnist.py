import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable
import numpy as np
import datetime
import os

# Function to convert 1 channel to 3 channels for MobileNetV2
    # See Section 3 for explanation on why we need to define a function here
    # Need to add the keras decorator otherwise Keras won't register it
        # If it's not registered, Keras cannot find it and deserialise it properly when loading the model
@register_keras_serializable()
def repeat_channels(img):
    return tf.repeat(img, 3, axis=-1)

# Define a directory to save models
models_dir = 'saved_models'
# Ensure the directory exists (create it if it doesn't)
os.makedirs(models_dir, exist_ok=True)

print(f"----- 1. Load Data (Fashion MNIST) -----")

# Load and preprocess the Fashion MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalise pixel values to be between 0 and 1
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include channel dimension for CNNs (28x28 images, 1 channel)
x_train_full = x_train_full.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Split training data into training and validation sets
x_train, x_val = x_train_full[:-5000], x_train_full[-5000:]
y_train, y_val = y_train_full[:-5000], y_train_full[-5000:]

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_val = keras.utils.to_categorical(y_val, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")    # x_train shape: (55000, 28, 28, 1), y_train shape: (55000, 10)
print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")            # x_val shape: (5000, 28, 28, 1), y_val shape: (5000, 10)
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")        # x_test shape: (10000, 28, 28, 1), y_test shape: (10000, 10)

print("-"*100)

print("----- 2. Define tf.data Pipelines -----")

# Define batch size (can use the one from the best hyperparams or a standard one)
BATCH_SIZE = 32     # Example

# Create tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Apply transformations
# For training data: cache (optional, if you want on-disk caching, uncomment .cache('filename')), shuffle, batch, prefetch
# For small datasets like Fashion MNIST, in-memory cache() is fine.
# For larger datasets, consider .cache('path/to/disk/cache_dir') before shuffle/batch.
# A larger buffer size leads to better shuffling but uses more memory. 
# For smaller datasets, can use len(x_train), but not for larger datasets as it'll exhaust the RAM (then use 1024, 5000, 10000, etc)
train_dataset = train_dataset.cache()                           # Caches data in memory after slicing, before shuffling/batching
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)) # Use full dataset size for best shuffle
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# For validation and test data: cache, batch, prefetch (no shuffling needed for evaluation)
val_dataset = val_dataset.cache()
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("\nDataset pipelines created:")
print(f"Train dataset element spec: {train_dataset.element_spec}")
# Train dataset element spec: (TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 10), dtype=tf.float64, name=None))
print(f"Validation dataset element spec: {val_dataset.element_spec}")
# Validation dataset element spec: (TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 10), dtype=tf.float64, name=None))
print(f"Test dataset element spec: {test_dataset.element_spec}")
# Test dataset element spec: (TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 10), dtype=tf.float64, name=None))

print("-"*100)

print("----- 3. Build the model -----")

# From the Random Search model, the best hyperparams were:
# - Units in Dense layer: 480
# - Dropout Rate: 0.3
# - Learning Rate: 0.000256153969803113 (approximately 2.56e-4)
# - Optimizer: adam

def build_optimized_model(hp_dense_units, hp_dropout_rate, hp_learning_rate):
    # Load the pre-trained MobileNetV2 model
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),    # Expected input shape for MobileNetV2
        include_top=False,          # Don't include the classifier head
        weights='imagenet'          # Load weights pre-trained on ImageNet
    )
    base_model.trainable = False    # Keep the pre-trained layers frozen

    # Create the functional model
    inputs = keras.Input(shape=(28, 28, 1))

    # --- Preprocessing for MobileNetV2 (as in original build_model) ---
    # MobileNetV2 expects 3-channel input
    # Convert 1-channel (grayscale) to 3-channels (simulated RGB) by repeating the channel
    # x = layers.Lambda(lambda img: tf.repeat(img, 3, axis=-1), output_shape=(28, 28, 3))(inputs)   
        # This was the previous code attempted, but when keras loads the model, it doesn't recognise 'tf' and, thus, gives a NameError
        # We, therefore, added a function at the top of the code after the imports and will use it here
    x = layers.Lambda(repeat_channels, output_shape=(28, 28, 3))(inputs)    # Need to specify output_size, otherwise, during rebuilding, Keras won't be able to infer the Lambda output shape
    # Resize from (28, 28, 3) to (96, 96, 3)
    x = layers.Resizing(96, 96)(x)

    x = base_model(x, training=False) # Run through the frozen base model

    # Add a Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    # --- Tuned Custom Classification Head ---
    # Use the best 'units' value
    x = layers.Dense(hp_dense_units, activation='relu')(x)
    # Use the best 'dropout_rate' value
    x = layers.Dropout(hp_dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # --- Tuned Optimiser and Learning Rate ---
    # Use the best 'learning_rate' as initial_learning_rate for the scheduler
    # And use the best 'optimiser' (Adam in this case)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp_learning_rate,
        decay_steps=1000, # A common setting for decay. You can adjust this.
                          # For example, if you have 48000 samples and batch_size=32, one epoch is 1500 steps.
                          # decay_steps=1500 * 5 would mean decay every 5 epochs.
        decay_rate=0.9,   # Decay LR by 10%
        staircase=True
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule) 

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Instantiate the model with best hyperparameters from Random Search
best_model = build_optimized_model(
    hp_dense_units=480,
    hp_dropout_rate=0.3,
    hp_learning_rate=0.000256153969803113 # Or 2.56e-4
)

best_model.summary()

# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer_1 (InputLayer)           │ (None, 28, 28, 1)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lambda (Lambda)                      │ (None, 28, 28, 3)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ resizing (Resizing)                  │ (None, 96, 96, 3)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ mobilenetv2_1.00_96 (Functional)     │ (None, 3, 3, 1280)          │       2,257,984 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ global_average_pooling2d             │ (None, 1280)                │               0 │
# │ (GlobalAveragePooling2D)             │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 480)                 │         614,880 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 480)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 10)                  │           4,810 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 2,877,674 (10.98 MB)
#  Trainable params: 619,690 (2.36 MB)
#  Non-trainable params: 2,257,984 (8.61 MB)


print("-"*100)

print("----- 4. Define Callbacks -----")

# Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1
)

# Model Checkpoint Callback
# Save the entire model in the new .keras format
# Construct the full path for the checkpoint file
checkpoint_filepath = os.path.join(models_dir, 'optimized_cnn_best_model.keras')

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# TensorBoard Callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_optimised_run"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,   # Log histograms every epoch
    write_graph=True,   # Write model graph
    update_freq='epoch' # Update logs after each epoch
)

# Combine all callbacks
callbacks = [early_stopping, model_checkpoint, tensorboard_callback]

print("-"*100)

print("----- 5. Train the Model -----")

EPOCHS = 50     # Setting a higher number of epochs, as EarlyStopping will stop it automatically

history = best_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)

"""
Epoch 1/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.8078 - loss: 0.5462 
Epoch 1: val_loss improved from inf to 0.29749, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.8079 - loss: 0.5461 - val_accuracy: 0.8900 - val_loss: 0.2975
Epoch 2/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.8930 - loss: 0.2970  
Epoch 2: val_loss improved from 0.29749 to 0.28158, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 57s 33ms/step - accuracy: 0.8930 - loss: 0.2970 - val_accuracy: 0.8946 - val_loss: 0.2816
Epoch 3/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.9107 - loss: 0.2462  
Epoch 3: val_loss improved from 0.28158 to 0.26282, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 56s 33ms/step - accuracy: 0.9107 - loss: 0.2462 - val_accuracy: 0.8996 - val_loss: 0.2628
Epoch 4/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.9162 - loss: 0.2248  
Epoch 4: val_loss improved from 0.26282 to 0.25671, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 57s 33ms/step - accuracy: 0.9162 - loss: 0.2248 - val_accuracy: 0.9030 - val_loss: 0.2567
Epoch 5/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9268 - loss: 0.2010  
Epoch 5: val_loss improved from 0.25671 to 0.24773, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 58s 33ms/step - accuracy: 0.9268 - loss: 0.2010 - val_accuracy: 0.9096 - val_loss: 0.2477
Epoch 6/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.9325 - loss: 0.1884  
Epoch 6: val_loss did not improve from 0.24773
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 57s 33ms/step - accuracy: 0.9325 - loss: 0.1884 - val_accuracy: 0.9036 - val_loss: 0.2560
Epoch 7/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.9364 - loss: 0.1762  
Epoch 7: val_loss did not improve from 0.24773
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 57s 33ms/step - accuracy: 0.9364 - loss: 0.1762 - val_accuracy: 0.9090 - val_loss: 0.2485
Epoch 8/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9417 - loss: 0.1593  
Epoch 8: val_loss improved from 0.24773 to 0.24485, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 58s 34ms/step - accuracy: 0.9417 - loss: 0.1593 - val_accuracy: 0.9084 - val_loss: 0.2448
Epoch 9/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9447 - loss: 0.1522  
Epoch 9: val_loss improved from 0.24485 to 0.24468, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9447 - loss: 0.1522 - val_accuracy: 0.9098 - val_loss: 0.2447
Epoch 10/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9513 - loss: 0.1432  
Epoch 10: val_loss improved from 0.24468 to 0.24459, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9513 - loss: 0.1432 - val_accuracy: 0.9082 - val_loss: 0.2446
Epoch 11/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9541 - loss: 0.1367  
Epoch 11: val_loss improved from 0.24459 to 0.24176, saving model to saved_models/optimized_cnn_best_model.keras
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9541 - loss: 0.1367 - val_accuracy: 0.9134 - val_loss: 0.2418
Epoch 12/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9550 - loss: 0.1315  
Epoch 12: val_loss did not improve from 0.24176
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 58s 34ms/step - accuracy: 0.9550 - loss: 0.1315 - val_accuracy: 0.9132 - val_loss: 0.2427
Epoch 13/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9560 - loss: 0.1260  
Epoch 13: val_loss did not improve from 0.24176
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9560 - loss: 0.1260 - val_accuracy: 0.9132 - val_loss: 0.2419
Epoch 14/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9558 - loss: 0.1247  
Epoch 14: val_loss did not improve from 0.24176
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9558 - loss: 0.1247 - val_accuracy: 0.9130 - val_loss: 0.2426
Epoch 15/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9593 - loss: 0.1209  
Epoch 15: val_loss did not improve from 0.24176
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9593 - loss: 0.1209 - val_accuracy: 0.9112 - val_loss: 0.2438
Epoch 16/50
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.9601 - loss: 0.1179  
Epoch 16: val_loss did not improve from 0.24176
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.9601 - loss: 0.1179 - val_accuracy: 0.9134 - val_loss: 0.2422
Epoch 16: early stopping
"""

# Load the best model saved by ModelCheckpoint for evaluation
print(f"\nLoading the best model from {checkpoint_filepath} for final evaluation...")
loaded_best_model = keras.models.load_model(checkpoint_filepath, safe_mode=False)   
# Adding safe_mode=False due to the lambda layer above
# When rebuilding, Keras would execute this arbitrary code, which could potentially come from an untrusted source
# As this is our code, which we know if safe, we can add safe_mode=False

# Evaluate the loaded best model on the test set
print("\nEvaluating the loaded best model on the test set:")
test_loss, test_accuracy = loaded_best_model.evaluate(test_dataset)
# Evaluating the loaded best model on the test set:
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 30ms/step - accuracy: 0.9026 - loss: 0.2542 
print(f"Test Loss: {test_loss:.4f}")            # 0.2543
print(f"Test Accuracy: {test_accuracy:.4f}")    # 0.9079

"""
After the script finishes, open terminal and navigate to the root of the project directory (where the logs folder will be created).
Run TensorBoard to visualize the training: 

`tensorboard --logdir logs/fit`

Go to the URL provided by TensorBoard (usually http://localhost:6006) to explore the different tabs (Scalars, Graphs, Histograms).
"""

"""
Due to differences in model performance per run, multiple runs will be carried out to assess model efficieny

The variation between runs can be due to:
1. Shuffle
    The data samples presented are different each run
    Weights are updated based on these batches
    Different batch order, therefore, leads to a different sequence of weight updates
2. Weight initialisation
    Initial weights are randomly set (although from a specific distribution, the actual numbers will be different each time)
    This can be avoided by setting a global seed
3. Dropout layers
    The number of neurons 'dropped' are random for each epoch and each run
4. The optimiser's internal state
    Optimisers like adam maintains two internal "state variables" for each trainable parameter (weight and bias in this model):
        - First Moment (Mean of Gradients, m): 
            This is an exponentially decaying average of past gradients. 
            It's similar to momentum and helps to smooth out the gradient updates.
        - Second Moment (Uncentered Variance of Gradients, v): 
            This is an exponentially decaying average of past squared gradients. 
            This gives the optimizer information about the "roughness" or scale of the gradients for each parameter.
        Adam uses the first moment (m) to guide the direction of the update.
        Adam uses the second moment (v) to scale the learning rate for each parameter.
            If a parameter has historically had large gradients (v is large), its effective learning rate will be reduced. 
            If it's had small gradients (v is small), its effective learning rate will be increased.
    The evolution of these states is influenced by the sequence of gradients received
        This depends on the batch order
5. GPU Non-determinism (not so common for smaller models)
    Low-level floating point operations can lead to tiny variations which compund over many epochs

Model scores per run:

Run     | Test Accuracy     | Test Loss     | Number of Epochs Used
--------|-------------------|---------------|-----------------------
1       | 0.9079            | 0.2543        | 16
2       | 0.9076            | 0.2590        | 17
3       | 0.9056            | 0.2559        | 17
4       | 0.9080            | 0.2534        | 17
5       | 0.9065            | 0.2555        | 17
6       | 0.9052            | 0.2583        | 14

Model Performance (after 6 runs): 0.9068 ± 0.001109

Overall:
- All independent runs performed similarly to each other
- The model performed well overall
    - However, inspecting the logs suggests possible overfitting due to discrepancies between the rate of increase and decline for accuracy and loss, repsectively, for the training and validation sets

Ways to overcome overfitting:
1. More data
2. Data Augmentatio
3. Regularisation Techniques (L1/L2 Regularisation)
4. Dropout
5. Early stopping
6. Batch normalisation
7. Model simplification
8. Hyperparameter Tuning
"""
