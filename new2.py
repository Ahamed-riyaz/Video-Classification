import tensorflow as tf
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import TensorBoard, EarlyStopping

frame_width, frame_height = 64, 64

tf.config.run_functions_eagerly(True)

# Load video frames and labels, preprocess, and create a dataset
def preprocess_video_frame(frame):
    # Preprocess frame: resize, normalize, etc.
    processed_frame = cv2.resize(frame, (frame_width, frame_height))
    processed_frame = processed_frame / 255.0
    return processed_frame

video_paths_cat = []
video_paths_dog = []

cat_dir = 'dataset/cat/'
dog_dir = 'dataset/dog/'
# fight-cat, no-dog
# List all video files (both .avi and .mp4) in the cat directory
for filename in os.listdir(cat_dir):
    if filename.endswith('.avi') or filename.endswith('.mp4'):
        video_paths_cat.append(os.path.join(cat_dir, filename))

# List all video files (both .avi and .mp4) in the dog directory
for filename in os.listdir(dog_dir):
    if filename.endswith('.avi') or filename.endswith('.mp4'):
        video_paths_dog.append(os.path.join(dog_dir, filename))

video_frames = []
labels = []

for video_path in video_paths_cat:
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_video_frame(frame)
        video_frames.append(preprocessed_frame)
        labels.append(1)  # 1 represents cat class

for video_path in video_paths_dog:
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_video_frame(frame)
        video_frames.append(preprocessed_frame)
        labels.append(0)  # 0 represents dog class

video_frames = np.array(video_frames)
labels = np.array(labels)

# video_frames = [
#     [frame_1_video_1, frame_2_video_1, ..., frame_n_video_1],
#     [frame_1_video_2, frame_2_video_2, ..., frame_n_video_2],
#     ...
#     [frame_1_video_m, frame_2_video_m, ..., frame_n_video_m]
# ]
# labels = [label_video_1, label_video_2, ..., label_video_m]

# this tensor slices will join like this 
#  create a dataset where each element is a tuple containing a frame sequence and its corresponding label

# [    ([frame_1_video_1, frame_2_video_1, ..., frame_n_video_1], label_video_1),
#     ([frame_1_video_2, frame_2_video_2, ..., frame_n_video_2], label_video_2),
#     ...
#     ([frame_1_video_m, frame_2_video_m, ..., frame_n_video_m], label_video_m)
# ]

video_dataset = tf.data.Dataset.from_tensor_slices((video_frames, labels))

# Split the dataset into train, validation, and test sets
train_size = int(len(video_frames) * 0.7)
val_size = int(len(video_frames) * 0.2)
test_size = len(video_frames) - (train_size + val_size)

train = video_dataset.take(train_size)
remaining_data = video_dataset.skip(train_size)
val = remaining_data.take(val_size)
test = remaining_data.skip(val_size).take(test_size)

# Define and compile your model
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3,),1,activation ='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3,),1,activation ='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(train, epochs=15, validation_data=val)

video_path_to_predict = 'cat.avi'  # Replace with the path to the video you want to predict
cap = cv2.VideoCapture(video_path_to_predict)
out = cv2.VideoWriter('output_' + os.path.basename(video_path_to_predict), cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    preprocessed_frame = preprocess_video_frame(frame)
    prediction = model.predict(np.expand_dims(preprocessed_frame, 0))

    if prediction < 0.5:
        label = "dog"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        label = "cat"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    out.write(frame)

cap.release()
out.release()

print("Video prediction and processing completed.")
