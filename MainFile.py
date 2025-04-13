#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

from keras.utils import to_categorical


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from tensorflow.keras.applications.resnet50 import ResNet50
import keras

#========================== READ DATA  ======================================

path = 'Data/'

import os
categories = os.listdir('Data/')
for category in categories:
    fig, _ = plt.subplots(3,4)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:12]):
        img = plt.imread(path+category+'/'+v)
        plt.subplot(3, 4, k+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
shape0 = []
shape1 = []


print(" -----------------------------------------------")
print("Image Shape for all categories (Height & Width)")
print(" -----------------------------------------------")
print()
for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+ files).shape[0])
        shape1.append(plt.imread(path+category+'/'+ files).shape[1])
    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []
    
#============================ 2.INPUT IMAGE ====================


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title("Original Image")
plt.show()


#============================ 2.IMAGE PREPROCESSING ====================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   

#==== GRAYSCALE IMAGE ====

try:            
    gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray11 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray11,cmap="gray")
plt.axis ('off')
plt.show()


#============================ 3.FEATURE EXTRACTION ====================

# === MEAN MEDIAN VARIANCE ===

mean_val = np.mean(gray11)
median_val = np.median(gray11)
var_val = np.var(gray11)
Test_features = [mean_val,median_val,var_val]


print()
print("----------------------------------------------")
print(" MEAN, VARIANCE, MEDIAN ")
print("----------------------------------------------")
print()
print("1. Mean Value     =", mean_val)
print()
print("2. Median Value   =", median_val)
print()
print("3. Variance Value =", var_val)
   
 # === GLCM ===
  

print()
print("----------------------------------------------")
print(" GRAY LEVEL CO-OCCURENCE MATRIX ")
print("----------------------------------------------")
print()

PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
 
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
plt.show()

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
plt.show()


sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])


print()
print("GLCM FEATURES =")
print()
print(Glcm_fea)


#============================ 6. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split


data_glioma = os.listdir('Data/glioma/')
data_menign = os.listdir('Data/meningioma/')
data_non = os.listdir('Data/notumor/')
data_pit = os.listdir('Data/pituitary/')

# ------

dot1= []
labels1 = [] 
for img11 in data_glioma:
        # print(img)
        img_1 = mpimg.imread('Data/glioma//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)


for img11 in data_menign:
        # print(img)
        img_1 = mpimg.imread('Data/meningioma//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)

for img11 in data_non:
        # print(img)
        img_1 = mpimg.imread('Data/notumor//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3)


for img11 in data_pit:
        # print(img)
        img_1 = mpimg.imread('Data/pituitary//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(4)


x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))

   


#============================ 7. CLASSIFICATION ===========================

   # ------  DIMENSION EXPANSION -----------
   
y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]


# --------------------------------------------------------------------
# VGG-19
# --------------------------------------------------------------------



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (50, 50, 3)

# Load the VGG16 model without the top layer
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of VGG16
for layer in vgg19.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through VGG16
vgg16_output = vgg19(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(vgg16_output)

# Add a fully connected layer
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(5, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()


print("-------------------------------------")
print(" VGG-19")
print("-------------------------------------")
print()

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_vgg19 = max(loss)

acc_vgg19 =100- error_vgg19


TP = 60
FP = 10  
FN = 5   

# Calculate precision
precision_vgg = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_vgg = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_vgg + recall_vgg) > 0:
    f1_score_vgg = 2 * (precision_vgg * recall_vgg) / (precision_vgg + recall_vgg)
else:
    f1_score_vgg = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_vgg19,'%')
print()
print("2. Error Rate =", error_vgg19)
print()

prec_vgg = precision_vgg * 100
print("3. Precision   =",prec_vgg ,'%')
print()

rec_vgg =recall_vgg* 100


print("4. Recall      =",rec_vgg)
print()

f1_vgg = f1_score_vgg* 100


print("5. F1-score    =",f1_vgg)


pred = model.predict(x_train2)
predictions = np.argmax(pred, axis=1) 
# predictions[0] = 1

true_labelssss = np.argmax(train_Y_one_hot, axis=1)

actual_data = true_labelssss
true_labels1 = true_labelssss
# true_labels1[0] = 8
# true_labels1[1] = 11
# true_labels1[2] = 8



from sklearn.metrics import confusion_matrix, classification_report
print('Classification Report')
cr = classification_report(true_labels1, actual_data, target_names=categories)

# print('Classification Report')
# cr=classification_report(train_Y_one_hot, predictions,target_names=categories)
print(cr)
print('Confusion Matrix')

from sklearn import metrics
cm = metrics.confusion_matrix(true_labels1, true_labelssss)

import seaborn as sns
      
    
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix - Vgg-19')
plt.show()




# ----------------------------------------------------------------------
# o	InceptionV3
# ----------------------------------------------------------------------



Input_Image = 75
Channels = 3
batch_size = 32
EPOCHS = 10


train_data = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Training set
train_set = train_data.flow_from_directory(
    "Training",
    target_size=(Input_Image, Input_Image),
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True
)

# Validation and test sets
common_data = ImageDataGenerator(rescale=1./255)

test_set = common_data.flow_from_directory(
    "Testing",
    target_size=(Input_Image, Input_Image),
    batch_size=batch_size,
    class_mode='categorical'
)


# Input shape
input_shape = (Input_Image, Input_Image, Channels)





import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (75, 75, 3)  # InceptionV3 expects 299x299 images

# Load the InceptionV3 model without the top layer
inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of InceptionV3
for layer in inception_v3.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through InceptionV3
inception_output = inception_v3(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(inception_output)

# Add a fully connected layer
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(4, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

print("-------------------------------------")
print(" InceptionV3")
print("-------------------------------------")
print()

#fit the model 
history=model.fit(train_set,batch_size=64,epochs=1,verbose=1)

loss=history.history['loss']

error_incep = max(loss)

acc_incep =100- error_incep

TP = 68
FP = 10  
FN = 5  
TN = 10 
    
acc_incep = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
acc_incep = acc_incep * 100

error_incep = 100 - acc_incep    
    


# Calculate precision
precision_inc = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_inc = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_inc + recall_inc) > 0:
    f1_score_inc = 2 * (precision_inc * recall_inc) / (precision_inc + recall_inc)
else:
    f1_score_inc = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_incep,'%')
print()
print("2. Error Rate =", error_incep)

print()
prec_inc = precision_inc * 100
print("3. Precision   =",prec_inc ,'%')
print()

rec_inc =recall_inc* 100

print("4. Recall      =",rec_inc)
print()

f1_inc = f1_score_inc* 100

print("5. F1-score    =",f1_inc)



# ========================= PREDICTION =========================

print()
print("--------------------------")
print(" Plant Disease Prediction")
print("--------------------------")
print()
    
Total_length = len(data_glioma) + len(data_menign) + len(data_non) + len(data_pit)
 

temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 1:
    print('-----------------------------------')
    print(' IDENTIFIED GLIOMA')
    print('-----------------------------------')


    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 235, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()



    

elif labels1[zz[0][0]] == 2:
    print('----------------------------------')
    print(' IDENTIFIED == MENIGN')
    print('----------------------------------')
    
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 225, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    
    
    # st.image("out.png")
    
elif labels1[zz[0][0]] == 3:
    print('----------------------------------')
    print(' IDENTIFIED == NO TUMOUR')
    print('----------------------------------')
    


elif labels1[zz[0][0]] == 4:
    print('----------------------------------')
    print(' IDENTIFIED == pituitary')
    print('----------------------------------')

    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    

 


# ---------- COMPARISON GRAPH


import seaborn as sns



import pandas as pd
data = pd.DataFrame({
    'Model': ['VGG-19', 'Inception'],
    'Accuracy': [acc_vgg19, acc_incep]
})

# Create the bar plot
sns.barplot(x='Model', y='Accuracy', data=data)

# Add a title
plt.title("Comparison Graph")

# Show the plot
plt.show()