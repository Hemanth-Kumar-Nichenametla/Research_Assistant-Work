import tensorflow as tf
import keras
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
tf.random.set_seed(42)

# Define the image size
IMAGE_SIZE = [224, 224]

# Set the paths to the training and validation data
train_path = 'chest_xray/train'
valid_path = 'chest_xray/test'

# Import the VGG16 model
vgg = VGG16(input_shape=IMAGE_SIZE+[3], 
            weights='imagenet', 
            include_top=False)

# Freeze all layers in the VGG16 model
for layer in vgg.layers:
    layer.trainable = False

# Get the folders in the training data directory
folders = glob('chest_xray/train/*')
# Add a flatten layer and a dense layer to the VGG16 model
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
model.summary()

# Compile the model
model.compile(
 loss='categorical_crossentropy',
 optimizer='adam',
 metrics=['accuracy']
)

# Create image data generators for the training and validation data
train_datagen = ImageDataGenerator(rescale = 1./255,
                  shear_range = 0.2,
                  zoom_range = 0.2,
                  horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Generate training and validation data
training_set = train_datagen.flow_from_directory('chest_xray/train',
                         target_size = (224, 224),
                         batch_size = 10,
                         class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('chest_xray/test',
                      target_size = (224, 224),
                      batch_size = 10,
                      class_mode = 'categorical')

# Train the model
model.fit(
 training_set,
 validation_data=test_set,
 epochs=1,
 steps_per_epoch=len(training_set),
 validation_steps=len(test_set)
)

img=image.load_img('chest_xray/test/Normal/IM-0001-0001.jpeg',target_size=(224,224))



x=image.img_to_array(img)

x=np.expand_dims(x, axis=0)

img_data=preprocess_input(x)

classes=model.predict(img_data)

result=int(classes[0][0])

if result==0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")





# Initialize an empty list to store true and predicted labels
true_labels = []
predicted_labels = []

# Loop through the test set and predict labels
for i in range(len(test_set)):
    batch = test_set[i]
    images, labels = batch
    predictions = model.predict(images)

    # Convert one-hot encoded labels to class labels
    true_labels.extend(np.argmax(labels, axis=1))
    predicted_labels.extend(np.argmax(predictions, axis=1))

true_labels

type(true_labels)

predicted_labels

len(predicted_labels)

len(true_labels)

t_labels=np.array(true_labels)

p_labels=np.array(predicted_labels)

type(t_labels)

type(p_labels)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix as a heatmap
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(np.arange(len(conf_matrix)) + 0.5, ['Class 0', 'Class 1'])  # Modify class labels accordingly
plt.yticks(np.arange(len(conf_matrix)) + 0.5, ['Class 0', 'Class 1'])  # Modify class labels accordingly
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Finding precision and recall
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy   :", accuracy)
precision = precision_score(true_labels, predicted_labels)
print("Precision :", precision)
recall = recall_score(true_labels, predicted_labels)
print("Recall    :", recall)
F1_score = f1_score(true_labels, predicted_labels)
print("F1-score  :", F1_score)

# Save the entire model as a binary file
#joblib.dump(model, 'vgg16_pneumonia_model.joblib')

model.save('/Users/hemu/Desktop/Updated_work_on_Deep_Learning/Research_work_on_Deep_Learning/Pneumonia/chest_xray/Pneumonia_models/VGG16_Cloud.h5')

import streamlit as st

# pip install streamlit

# Streamlit app
def main():
    st.title('VGG16 Pneumonia Image Classification')
    st.write('Upload an image for classification:')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make predictions when the user clicks a button
        if st.button('Classify'):
            prediction = predict(image)
            st.write('Prediction:', prediction)
if __name__ == '__main__':
    main()


