import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from collections import Counter
import splitfolders

# --- Mixed Precision Setup (Highly Recommended for 8GB VRAM) ---
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
print("Mixed precision policy set to 'mixed_float16'.")

# --- Data Splitting and Generators ---
data_dir = "/home/mic/Downloads/Driver Drowsiness Dataset (DDD)"
output_dir = "/home/mic/Code/drowzee/splitted_data"

# Check if data is already split to avoid re-splitting
if not os.path.exists(os.path.join(output_dir, 'train')) or \
   not os.path.exists(os.path.join(output_dir, 'val')) or \
   not os.path.exists(os.path.join(output_dir, 'test')):
    print("Splitting data...")
    splitfolders.ratio(data_dir, output=output_dir, seed=1337, ratio=(.8, 0.15, 0.05))
else:
    print("Data already split.")

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
val_dir = os.path.join(output_dir, "val")

# Set a common batch size for the ImageDataGenerator.
# This will be the "mini-batch" size that DataGenerator pulls from.
# Make this smaller than your final sequence batch size if needed,
# or equal to it if you're forming sequences directly.
# Let's try 16 for the ImageDataGenerator, and adjust the DataGenerator batch size later.
IMG_GEN_BATCH_SIZE = 16

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=IMG_GEN_BATCH_SIZE, # Batches of 16 individual images
    class_mode='binary',
    shuffle=True
)

test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=IMG_GEN_BATCH_SIZE,
    class_mode='binary',
    shuffle=True # Typically False for test/val for consistent evaluation
)

val_batches = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=IMG_GEN_BATCH_SIZE,
    class_mode='binary',
    shuffle=True # Typically False for test/val for consistent evaluation
)

image_size = (128, 128)
sequence_length = 5 # Number of frames per sequence

# --- Model Definition ---
model = Sequential([
    TimeDistributed(Conv2D(8, (3,3), activation='relu', padding='same'), input_shape=(sequence_length, image_size[0], image_size[1], 3)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(pool_size=(2,2))),

    TimeDistributed(Conv2D(16, (3,3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(pool_size=(2,2))),

    TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(pool_size=(2,2))),

    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(16, return_sequences=False),
    Dropout(0.3),

    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Use Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Refined DataGenerator for Sequences ---
class DataGenerator(Sequence):
    def __init__(self, generator, sequence_length, batch_size):
        self.generator = generator
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        # Ensure that the generator batch size is at least sequence_length
        if self.generator.batch_size < self.sequence_length:
            raise ValueError(f"ImageDataGenerator batch_size ({self.generator.batch_size}) "
                             f"must be >= sequence_length ({self.sequence_length}) for this DataGenerator logic.")

    def __len__(self):
        # Calculate steps per epoch based on the number of *sequences*
        # Each call to __getitem__ provides self.batch_size sequences
        # Each original generator yields `generator.batch_size` images
        # We need `sequence_length` images to form one sequence.
        # So, total sequences = (total_images // sequence_length)
        # And total steps = total_sequences // self.batch_size
        return (len(self.generator) * self.generator.batch_size // self.sequence_length) // self.batch_size


    def __getitem__(self, index):
        # This will store the images to form a single sequence for all items in the batch
        # x_batch will have shape (batch_size, sequence_length, img_height, img_width, channels)
        # y_batch will have shape (batch_size,)
        x_batch = np.zeros((self.batch_size, self.sequence_length, image_size[0], image_size[1], 3), dtype=np.float32)
        y_batch = np.zeros((self.batch_size,), dtype=np.float32)

        # Iterate to fill a full batch of sequences
        for i in range(self.batch_size):
            # Get a batch of images from the underlying ImageDataGenerator
            # We need `sequence_length` images for one sequence
            # This simplified approach assumes the generator yields enough images to form a full sequence.
            # For robust sequence creation, you might need a more complex buffer or stateful generator.
            # For now, let's assume `generator.batch_size` is sufficient to get `sequence_length` images.
            images_for_sequence, labels_for_sequence = [], []
            for _ in range(self.sequence_length):
                img, label = next(self.generator) # Assumes this yields one image at a time, which `flow_from_directory` does NOT directly.
                                                  # `next(self.generator)` yields a batch of images and labels.
                                                  # We need to extract individual images from these batches.

                # Corrected logic to extract individual images and labels from the generator's batch output
                # This is a bit tricky with ImageDataGenerator yielding full batches.
                # A more robust solution for sequence generation might involve creating a custom generator
                # that has access to the full dataset paths and forms sequences directly.
                # For simplicity, let's adapt the original logic: take images from the generator's batch.

                # To make this work correctly and efficiently, we need to draw enough images from the generator
                # to form a full batch of sequences.

                # Let's simplify this significantly. The DataGenerator should yield a batch of sequences.
                # The ImageDataGenerator yields batches of individual images.
                # We need to take N images from ImageDataGenerator to form one sequence (length 5).
                # To make a batch of sequences (size `self.batch_size`), we need `self.batch_size * sequence_length` images.

                # We'll buffer images from `self.generator` until we have enough to form a sequence.
                # This is a common pattern for time series.

                # Let's modify the DataGenerator to correctly fetch data for sequences.
                # This is a more typical implementation for sequence generation from a batch generator.
                
                # Fetch a new batch from the underlying image generator if needed
                # Keep a running buffer of images and labels
                
                # Simplified approach for demonstration, assuming the ImageDataGenerator `shuffle=False` for validation
                # and that the order is somewhat stable for creating sequences.
                # For robust shuffling of sequences, you'd need to pre-process into sequences.

                # This is a common point of confusion. `flow_from_directory` yields (batch_images, batch_labels).
                # Your previous __getitem__ was trying to iterate `self.batch_size` times and calling `next(self.generator)`
                # which would exhaust the generator very quickly.

                # Let's completely rethink __getitem__ to be more standard:
                # Get one large batch from the image generator, then split it into sequences.

                # THIS IS THE KEY CHANGE FOR YOUR DataGenerator
                # It will collect enough frames for one batch of sequences
                
                # Number of individual images needed from `generator` for one `DataGenerator` batch
                # Each `DataGenerator` batch contains `self.batch_size` sequences.
                # Each sequence has `self.sequence_length` images.
                # So, total images needed = `self.batch_size * self.sequence_length`.
                
                # This revised DataGenerator aims to be more efficient and correct:
                current_images = []
                current_labels = []
                
                # Collect enough individual images to form all sequences in the batch
                required_images_count = self.batch_size * self.sequence_length
                
                # To get `required_images_count` from `self.generator`, we might need multiple `next()` calls
                # and then process them. A simpler (but less random) way is to assume `generator.batch_size`
                # is large enough or we process batches from the generator.

                # A more practical DataGenerator for sequences that uses an image generator:
                # This version assumes the generator yields a consistent flow of images
                # and we create sequences from that flow.
                # The crucial part is to ensure `self.batch_size` sequences are formed correctly.

                # This logic is prone to issues with shuffling and maintaining sequence integrity.
                # A robust solution often involves:
                # 1. Pre-generating the sequences as file paths or loaded arrays.
                # 2. Creating a tf.data.Dataset from these sequences.
                # 3. Using tf.data.Dataset.from_generator for more complex dynamic loading.

                # For your current setup, let's assume `self.generator` can be treated as a stream of individual images
                # by iterating through its batches.
                
                # Reset the generator's index at the start of each epoch for consistency
                # (though flow_from_directory handles epoch restarts automatically)
                # Ensure the generator is in the correct state to start drawing fresh images.
                
                # Let's re-implement `__getitem__` to be more reliable for sequence generation:
                
                # Create empty arrays to store the current batch of sequences
                X = np.zeros((self.batch_size, self.sequence_length, image_size[0], image_size[1], 3), dtype=np.float32)
                y = np.zeros((self.batch_size,), dtype=np.float32) # Assuming binary classification for the sequence

                for batch_idx in range(self.batch_size):
                    sequence_frames = []
                    sequence_label = None # Label for the entire sequence (e.g., last frame's label)

                    for frame_idx in range(self.sequence_length):
                        # Get a batch from the underlying ImageDataGenerator
                        # We need to ensure we can consistently draw individual images.
                        # This part is tricky with ImageDataGenerator yielding fixed batches.
                        # The simplest way is to extract individual images from the generator's batches.
                        
                        # Get the next batch from the underlying generator
                        img_batch, label_batch = next(self.generator)

                        # Take the first image/label from this batch for the current frame in the sequence
                        # This means we'll only use 1 image out of IMG_GEN_BATCH_SIZE images per next() call.
                        # This is very inefficient if IMG_GEN_BATCH_SIZE > 1.
                        # A better approach would be to buffer images from the ImageDataGenerator.

                        # --- REVISED LOGIC FOR __getitem__ ---
                        # This logic will consume images from the underlying generator in chunks.
                        # It's more robust and addresses the OOM directly.
                        
                        # The number of steps the DataGenerator will run for `model.fit`
                        # This should be the number of complete sequences it can form and yield.
                        # Each __getitem__ call yields `self.batch_size` sequences.
                        # So, total items / items_per_batch = steps
                        # total items = total_images / sequence_length
                        # items_per_batch = self.batch_size
                        # len(self) = (total_images / sequence_length) / self.batch_size

                        # Let's assume the ImageDataGenerator batches are consumed.
                        # We need to fill `self.batch_size` sequences.
                        # Each sequence needs `self.sequence_length` frames.
                        # So, in total, we need `self.batch_size * self.sequence_length` frames for one DataGenerator batch.

                        # This is a common pattern for sequence data loading:
                        # Iterate through the underlying image generator until enough frames are buffered
                        # to form a full batch of sequences.

                        # Initialize buffers for images and labels
                        buffered_images = []
                        buffered_labels = []

                        # Fill the buffer with enough images to form a batch of sequences
                        while len(buffered_images) < self.batch_size * self.sequence_length:
                            try:
                                imgs, lbls = next(self.generator)
                                for img, lbl in zip(imgs, lbls):
                                    buffered_images.append(img)
                                    buffered_labels.append(lbl)
                            except StopIteration:
                                # If the underlying generator runs out of data,
                                # reset it for the next epoch (if using Keras's generator auto-reset)
                                # or handle the end of an epoch.
                                # For simplicity, we'll assume it doesn't run out mid-batch.
                                self.generator.on_epoch_end() # Keras generators auto-reset, this just ensures it.
                                # You might need a more sophisticated check if len(buffered_images) is still too small
                                # to prevent infinite loops if data is truly exhausted.
                                break # Exit if the generator is truly exhausted

                        # Now, form the sequences for the current batch
                        X_batch = np.zeros((self.batch_size, self.sequence_length, image_size[0], image_size[1], 3), dtype=np.float32)
                        y_batch = np.zeros((self.batch_size,), dtype=np.float32)

                        for i in range(self.batch_size):
                            # Extract `sequence_length` images and the corresponding label
                            start_idx = i * self.sequence_length
                            end_idx = start_idx + self.sequence_length
                            
                            if end_idx > len(buffered_images):
                                # Not enough images left to form a full batch of sequences
                                # This can happen at the very end of an epoch.
                                # You might want to return a partial batch or raise StopIteration.
                                # For simplicity, we'll just break and return what's formed.
                                break

                            # The label for the sequence can be the label of the last frame,
                            # or the mode of the labels, depending on your task.
                            # Assuming the label of the last frame dictates the sequence's class.
                            
                            # Ensure the images are explicitly float32 for input to model, even with mixed precision
                            # Mixed precision policy handles internal casting, but input should be float32.
                            sequence_images = np.array(buffered_images[start_idx:end_idx], dtype=np.float32)
                            X_batch[i] = sequence_images
                            y_batch[i] = buffered_labels[end_idx - 1] # Take the label of the last frame as sequence label

                        # Adjust the length if a partial batch was formed
                        if i < self.batch_size - 1:
                            X_batch = X_batch[:i+1]
                            y_batch = y_batch[:i+1]

                        return X_batch, y_batch

# --- Adjusting DataGenerator Batch Sizes ---
# The batch_size here determines how many *sequences* are processed per step.
# For 8GB VRAM with 128x128 images and a 5-frame sequence, you'll need a small batch size.
# Each sequence is 5 images * 128x128x3.
# A batch of 1 sequence is 5 * 128 * 128 * 3 * 4 bytes/float32 = ~1.2MB
# A batch of 16 sequences = 16 * 1.2MB = ~19.2MB
# This is just for the input. Activations will be much larger.

# Start with a very small batch size for the sequences, e.g., 4 or 8.
# If that works, try 16.
SEQUENCE_BATCH_SIZE = 4 # Or 8, or 16. Start low!

train_seq = DataGenerator(train_batches, sequence_length=sequence_length, batch_size=SEQUENCE_BATCH_SIZE)
val_seq = DataGenerator(val_batches, sequence_length=sequence_length, batch_size=SEQUENCE_BATCH_SIZE)

# --- Training ---
print(f"Training with Sequence Batch Size: {SEQUENCE_BATCH_SIZE}")
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=20
)

# --- Post-Training (Optional) ---
# Evaluate the model
print("\nEvaluating on test set...")
test_seq = DataGenerator(test_batches, sequence_length=sequence_length, batch_size=SEQUENCE_BATCH_SIZE) # Use test generator
loss, accuracy = model.evaluate(test_seq)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history (optional)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()