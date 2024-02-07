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

video_paths_fighting = []
video_paths_nofighting = []

fighting_dir = 'dataset/fights/'
nofighting_dir = 'dataset/nofight/'

# List all video files (both .avi and .mp4) in the fighting directory
for filename in os.listdir(fighting_dir):
    if filename.endswith('.avi') or filename.endswith('.mp4'):
        video_paths_fighting.append(os.path.join(fighting_dir, filename))

# List all video files (both .avi and .mp4) in the nofight directory
for filename in os.listdir(nofighting_dir):
    if filename.endswith('.avi') or filename.endswith('.mp4'):
        video_paths_nofighting.append(os.path.join(nofighting_dir, filename))

video_frames = []
labels = []

for video_path in video_paths_fighting:
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_video_frame(frame)
        video_frames.append(preprocessed_frame)
        labels.append(1)  # 1 represents fighting class

for video_path in video_paths_nofighting:
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_video_frame(frame)
        video_frames.append(preprocessed_frame)
        labels.append(0)  # 0 represents no fighting class

video_frames = np.array(video_frames)
labels = np.array(labels)

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

# Training loop
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
hist = model.fit(train, epochs=15, validation_data=val, callbacks=[early_stopping_callback])

video_path_to_predict = 'newfi17.avi'  # Replace with the path to the video you want to predict
cap = cv2.VideoCapture(video_path_to_predict)
out = cv2.VideoWriter('output_' + os.path.basename(video_path_to_predict), cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    preprocessed_frame = preprocess_video_frame(frame)
    prediction = model.predict(np.expand_dims(preprocessed_frame, 0))

    if prediction < 0.3:
        label = "fighting"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()

print("Video prediction and processing completed.")










































import tensorflow as tf
import os
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.utils import *
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.callbacks import TensorBoard, EarlyStopping
from keras.metrics import Precision,Recall,BinaryAccuracy
from keras.models import load_model


# # def extract_frames(video_path, output_dir, label,frame_skip=5):
# #     cap = cv2.VideoCapture(video_path)
# #     frame_count = 0
# #     saved_frame_count = 0

# #     while cap.isOpened():
# #         ret, frame = cap.read()

# #         if not ret:
# #             break

# #         if frame_count % frame_skip == 0:
# #             timestamp = int(time.time())  # Get the current timestamp
# #             frame_filename = f"{label}_{timestamp}_{saved_frame_count:04d}.jpg"
# #             frame_path = os.path.join(output_dir, frame_filename)
# #             print(f"Saving frame: {frame_path}")
# #             cv2.imwrite(frame_path, frame)
# #             saved_frame_count += 1

# #         frame_count += 1

# #     cap.release()

# # # Paths for input videos and output directories
# # # input_fight_dir = 'NEW/fight'
# # input_nofight_dir = 'Real Life Violence Dataset/noFight'
# # # output_fight_dir = 'sample2/fight'
# # output_nofight_dir = 'sample2/noFight'

# # # Create output directories if they don't exist
# # # os.makedirs(output_fight_dir, exist_ok=True)
# # os.makedirs(output_nofight_dir, exist_ok=True)

# # # Process fight videos
# # # fight_videos = [f for f in os.listdir(input_fight_dir) if f.endswith('.mp4') or f.endswith('.avi')]
# # # for video_filename in fight_videos:
# # #     video_path = os.path.join(input_fight_dir, video_filename)
# # #     output_dir = output_fight_dir
# # #     extract_frames(video_path, output_dir, 'fight',frame_skip=5)

# # # Process nofight videos
# # nofight_videos = [f for f in os.listdir(input_nofight_dir) if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.gif')]
# # for video_filename in nofight_videos:
# #     video_path = os.path.join(input_nofight_dir, video_filename)
# #     output_dir = output_nofight_dir
# #     extract_frames(video_path, output_dir, 'noFight',frame_skip=5)

# # print("Frame extraction completed.")


# cpus = tf.config.experimental.list_physical_devices('CPU')
# # for cpu in cpus:
# #     tf.config.experimental.set_memory_growth(cpu,True)
# # print(gpus)

# data_dir = 'sample2'
# # imgext = ['.jpg','.jpeg','.bmp','.png','.gif']
# print(os.listdir(data_dir))

# for img_cls in os.listdir(data_dir):
#     for img in os.listdir(os.path.join(data_dir, img_cls)):
#         image_path = os.path.join(data_dir, img_cls, img)
#         _, ext = os.path.splitext(img)
#         ext_lower = ext.lower()

#         if ext_lower in ['.jpg', '.jpeg','.gif']:
#             try:
#                 img = tf.io.decode_image(tf.io.read_file(image_path))
#             except tf.errors.InvalidArgumentError:
#                 # print(f"Removing corrupted image: {image_path}")
#                 os.remove(image_path)

# # tf.data.Dataset

# data = tf.keras.utils.image_dataset_from_directory('sample2',batch_size=8,image_size=(128, 128))
# # for x, y in data:
# #     print(x)
# #     print(y)
# # print(data)
# # data_itr = data.as_numpy_iterator()
# # print(data_itr)
# # batch = data_itr.next()
# # scaled = batch[0] / 255
# data = data.map(lambda x,y: (x/255,y))
# scaled_itr = data.as_numpy_iterator()
# batch = scaled_itr.next()

# print(batch[0].min())
# print(batch[0].max())
# fig,ax = plt.subplots(ncols=4,figsize=(20,20))
# for idx,img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(batch[1][idx])
# # plt.show()

# # class 0 fights,class 1 nofight
# # print("fghj",data)
# # print(len(data))
# # print(data.as_numpy_iterator().next())
# total_size = len(data)
# train_ratio = 0.7
# val_ratio = 0.2
# test_ratio = 0.1

# train_size = int(total_size * train_ratio)
# val_size = int(total_size * val_ratio)
# test_size = total_size - (train_size + val_size)

# # print(total_size,train_size, val_size, test_size)

# train = data.take(train_size)
# remaining_data = data.skip(train_size)
# val = remaining_data.take(val_size)
# test = remaining_data.skip(val_size).take(test_size)

# # print(train,val,test)
# # print(len(train),len(val),len(test))
# model = Sequential()

# model.add(Conv2D(16,(3,3,),1,activation ='relu',input_shape = (128,128,3)))
# model.add(MaxPooling2D())

# model.add(Conv2D(32,(3,3,),1,activation ='relu'))
# model.add(MaxPooling2D())

# model.add(Conv2D(16,(3,3,),1,activation ='relu'))
# model.add(MaxPooling2D())

# model.add(Flatten())

# model.add(Dense(128,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

# model.summary()

# logdir = 'logs'
# tensorflow_callbacks = TensorBoard(log_dir=logdir)
# # early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
# # callbacks_list = [tensorflow_callbacks,early_stopping_callback]
# hist = model.fit(train,epochs=10,validation_data=val,callbacks=[tensorflow_callbacks])

# # fig = plt.figure()
# # plt.plot(hist.history['loss'],color='teal',label='loss')
# # plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
# # fig.suptitle('Loss',fontsize=20)
# # plt.legend(loc="upper left")
# # plt.show()

# # fig = plt.figure()
# # plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
# # plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
# # fig.suptitle('Accuracy',fontsize=20)
# # plt.legend(loc="upper left")
# # plt.show()

# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()

# for batch in test.as_numpy_iterator():
#     x,y = batch
#     yhat = model.predict(x)
#     pre.update_state(y,yhat)
#     re.update_state(y,yhat)
#     acc.update_state(y,yhat)

# print(f'precision:{pre.result().numpy()},recall:{re.result().numpy()},binacc:{acc.result().numpy}')

# model.save(os.path.join('models','VNV128.h5'))
loadmodel = load_model('convlstm_model___Date_Time_2023_08_16__11_11_30___Loss_0.04559596627950668___Accuracy_0.9841269850730896.h5')
video_path = 'Basketball Court Fight Scene _ The Expendables (720p).mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get the video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object
output_path = 'output2.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

num_frames = 20
frame_list = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to match the model's spatial input dimensions
    resized_frame = cv2.resize(frame, (64, 64))

    # Add the resized frame to the list of frames
    frame_list.append(resized_frame)

    # When the required number of frames is collected, process them
    if len(frame_list) == num_frames:
        # Stack frames along a new axis to create the temporal dimension
        frames_for_prediction = np.stack(frame_list, axis=0)
        frames_for_prediction = frames_for_prediction / 255.0  # Normalize pixel values

        # Make prediction
        predictions = loadmodel.predict(np.expand_dims(frames_for_prediction, axis=0))
        max_prob_class = np.argmax(predictions)  # Get the class index with the maximum probability

        # Overlay the label on the last frame
        # label = "fighting" if max_prob_class == 0 and predictions[0, max_prob_class] >= 0.8 else "nofighting"
        # cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if max_prob_class == 0 and predictions[0, max_prob_class] >= 0.8:
            label = "fighting"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            pass


        # Write the frame to the output video
        out.write(frame)

        # Remove the oldest frame from the list to make space for the new one
        frame_list.pop(0)

cap.release()
out.release()
# cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Preprocess the frame
#     resize = tf.image.resize(frame, (64, 64))
#     # resize = cv2.resize(frame, (128, 128))
#     # resize = resize / 255.0

#     # Make prediction
#     prediction = loadmodel.predict(np.expand_dims(resize / 255, 0))
#     print(prediction)

#     # Overlay the label on the frame
#     # label = "fighting" if prediction < 0.5 else "nofighting"
#     if prediction < 0.3:
#         label = "fighting"
#         # print("fighting")
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     else:
#         # print("Nofighting")
#         pass

#     # Write the frame to the output video
#     out.write(frame)

#     # Display the frame
#     # cv2.imshow('Frame', frame)

#     # Exit the loop if 'q' is pressed
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break

# cap.release()
# out.release()
# cv2.destroyAllWindows()


#detection