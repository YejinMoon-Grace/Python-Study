import os
import sys

conda_prefix = os.environ.get('CONDA_PREFIX', '/home/gen2/miniconda3/envs/torch')


os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_prefix}"

import streamlit as st 
import tensorflow as tf 
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import numpy as np

st.title("CNN 모델 학습")

print ("------------model load-------------")
model = tf.keras.models.load_model('dog_breed.hdf5')
print ("---------master load----------")
labels_ohe_names = pd.read_pickle("master.pkl")

uploaded_file = st.file_uploader(
    "이미지 파일을 선택하세요", 
    type={'png', 'jpg', 'webp'})
    
    
if uploaded_file is not None:
    st.image(uploaded_file)
    image = Image.open(uploaded_file)
    dog_image =img_to_array(load_img(uploaded_file, target_size=(299,299))).astype('float32')

    # img_to_array(load_img("{}".format(target), target_size=(299,299))).astype('float32')
    dog_image = dog_image.reshape(1, 299, 299, 3)
    dog_image /= 255
    result = labels_ohe_names.columns[np.argmax(model.predict(dog_image))]
    print(result)
    st.markdown(result)
