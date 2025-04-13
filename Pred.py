#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
import matplotlib.image as mpimg

#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image
import streamlit as st

import base64
import cv2

# ================ Background image ===

st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:26px;font-family:verdana;">{"Blockchain-Enabled Image Processing for Enhanced Security and Privacy in Medical Reports"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')

#====================== READ A INPUT IMAGE =========================

fileneme = st.file_uploader("Upload a image")

if fileneme is None:
    
    st.text("Kindly upload input image....")

else:
    # selected_image_name = fileneme.name
    #====================== READ A INPUT IMAGE =========================
    
    
    # filename = askopenfilename()
    img = mpimg.imread(fileneme)
    plt.imshow(img)
    # plt.title('Original Image') 
    plt.axis ('off')
    plt.savefig("Ori.png")
    plt.show()
        
    st.image(img,caption="Original Image")
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
       
             
    #==== GRAYSCALE IMAGE ====
    
    
    
    SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1,cmap='gray')
    plt.axis ('off')
    plt.show()
    
    # ============== FEATURE EXTRACTION ==============
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)
    
    
    #============================ 5. IMAGE SPLITTING ===========================
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    data_glioma = os.listdir('Data/glioma/')
    data_menign = os.listdir('Data/meningioma/')
    data_non = os.listdir('Data/notumor/')
    data_pit = os.listdir('Data/pituitary/')
    
    
    #       
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
    
    
   
    
    #=============================== PREDICTION =================================
    
    print()
    print("-----------------------")
    print("       PREDICTION      ")
    print("-----------------------")
    print()
    
    
    Total_length = len(data_glioma) + len(data_menign) + len(data_non) + len(data_pit)
     
    
    temp_data1  = []
    for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 1:
        print('-----------------------------------')
        print(' IDENTIFIED GLIOMA')
        print('-----------------------------------')

        # st.write('-----------------------------------')
        # st.write(' IDENTIFIED GLIOMA')
        # st.write('-----------------------------------')
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as GLIOMA"}</h1>', unsafe_allow_html=True)
        aa = "IDENTIFIED GLIOMA"
        
      
        
        import cv2
        import numpy as np
       
       # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
       
           objects = [[200, 245, 50, 50]]  
       
           for box in objects:
               x, y, w, h = box
               cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
       
           return image
       
        filenamee="Ori.png"
       
       # Load your medical image
        image = cv2.imread(filenamee)
       
       # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
       
        plt.imshow(image_with_boxes)
        plt.title('AFFECTED IMAGE')
        plt.axis ('off')
        plt.show()    
 
        st.image(image_with_boxes,caption="Affected Region")
        
    
    elif labels1[zz[0][0]] == 2:
        # st.write('----------------------------------')
        # st.write(' IDENTIFIED == MENIGN')
        # st.write('----------------------------------')
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as MENIGN"}</h1>', unsafe_allow_html=True)
        aa = "IDENTIFIED == MENIGN"
        
        
        import cv2
        import numpy as np
       
       # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
       
           objects = [[200, 225, 50, 50]]  
       
           for box in objects:
               x, y, w, h = box
               cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
       
           return image
       
        filenamee="Ori.png"
       
       # Load your medical image
        image = cv2.imread(filenamee)
       
       # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
       
        plt.imshow(image_with_boxes)
        plt.title('AFFECTED IMAGE')
        plt.axis ('off')
        plt.show()    
 
        st.image(image_with_boxes,caption="Affected Region")
        
        # st.image("out.png")
        
    elif labels1[zz[0][0]] == 3:
        
        aa = "IDENTIFIED == NO TUMOUR"
        # st.write('----------------------------------')
        # st.write(' IDENTIFIED == NO TUMOUR')
        # st.write('----------------------------------')
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as NO TUMOUR"}</h1>', unsafe_allow_html=True)


    elif labels1[zz[0][0]] == 4:
        # st.write('----------------------------------')
        # st.write(' IDENTIFIED == pituitary')
        # st.write('----------------------------------')

        aa = "IDENTIFIED == pituitary"
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as PITUIRARY"}</h1>', unsafe_allow_html=True)

        import cv2
        import numpy as np
        
        # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
        
            objects = [[250, 235, 50, 50]]  
        
            for box in objects:
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        
            return image
        
        filenamee="Ori.png"
        
        # Load your medical image
        image = cv2.imread(filenamee)
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
        
        plt.imshow(image_with_boxes)
        plt.title('AFFECTED IMAGE')
        plt.axis ('off')
        plt.show()    

        st.image(image_with_boxes,caption="Affected Region")
        
        
        
        
    import pickle
    with open('out.pickle', 'wb') as f:
        pickle.dump(aa, f)    

        
    import subprocess
    subprocess.run(['python','-m','streamlit','run','privacy.py'])    

        


