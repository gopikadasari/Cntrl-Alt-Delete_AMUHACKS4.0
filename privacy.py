#======================== IMPORT PACKAGES ===========================


import streamlit as st
import joblib 
import base64


#========================  BACKGROUND IMAGE ===========================


st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:26px;font-family:verdana;">{"Blockchain-Enabled Image Processing for Enhanced Security and Privacy in Medical Reports"}</h1>', unsafe_allow_html=True)
st.write("-------------------------------------------")



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


#================================= ENCRYPTION ===========================

#=== LIST TO STR ====


# encry_pass = input("Enter Password for encryption ")

encry_pass = st.text_input("Enter Password for encryption ",type="password")

butt1=st.button("Encrypt")

if butt1:
    import pickle
    with open('out.pickle', 'rb') as f:
        final_res = pickle.load(f)

    listToStr = ' '.join([str(elem) for elem in final_res])
    
    import rsa
    
    publicKey, privateKey = rsa.newkeys(512)
    
    with open('keyy.pickle', 'wb') as f:
        pickle.dump(privateKey, f)
    
    Encrypt = rsa.encrypt(listToStr.encode(),publicKey)

    with open('encryptt.pickle', 'wb') as f:
        pickle.dump(Encrypt, f)
        
        
        
    print("-------------------------------------------")
    print("Step 4 ------> Encryption  ")
    print("-------------------------------------------")
    print()
    
    print("The Encrypted String is ")
    print()
    print(Encrypt)
    print()
    
    st.success("Encrypted Succesfully !!!")


    import os.path
    
    save_path = 'Cloud/Encrypt/'
    
    completeName = os.path.join(save_path, "enc_info.txt")         
    
    file1 = open(completeName, "wb")
    
    file1.write(Encrypt)
    
    file1.close()
    
    st.success("Encrypted data stored in cloud Succesfully !!!")


    
#=============================== DECRYPTION ===========================


# decrypty_pass = input("Enter Password for Decryption = ")

decrypty_pass = st.text_input("Enter Password for Decryption ",type="password")

butt2 = st.button("Decrypt")
# # encry_pass

if butt2:

    if str(encry_pass)==str(decrypty_pass):
        import pickle
    #     st.text("matched")
        with open('encryptt.pickle', 'rb') as f:
            Encrypt = pickle.load(f)
            
        with open('keyy.pickle', 'rb') as f:
            privateKey = pickle.load(f)   
            
        import rsa
        
        Decrypt = rsa.decrypt(Encrypt, privateKey).decode()
    
        import os.path
        
        save_path = 'Cloud/Decrypt/'
        
        completeName = os.path.join(save_path, "dec_info.txt")         
        
        file1 = open(completeName, "w")
        
        file1.write(Decrypt)
        
        file1.close()
        
            
        st.write("-------------------------------------------")
        st.write("Decryption ")
        st.write("-------------------------------------------")
        print()
        st.write("The decrypted string : \n\n",Decrypt)
      
        
    else:
        st.warning("Error!!!, Password is Not Matched")
        print()
        
        import os.path
        
        save_path = 'Cloud/Encrypt/'
        
        completeName = os.path.join(save_path, "enc_info.txt")         
        
        file1 = open(completeName, "wb")
        
        file1.write(Encrypt)
    
        file1.close()
        
        

butt3=st.button("PRIVACY BLOCK CHAIN ")   

if butt3:

    import subprocess
    subprocess.run(['python','-m','streamlit','run','BlockChain.py'])
        
        
        
        
        
        
        
        
        