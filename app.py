import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
from tensorflow.keras.models import load_model, Model
from pickle import load
from tensorflow.keras.applications import DenseNet201
from PIL import Image



# Load the saved model and tokenizer
caption_model = load_model("image_cap_model.h5")
#tokenizer = load_tokenizer("tokenizer.pkl")  # Assuming you saved the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
img_size=224

#Define feature extraction function
def extract_feature(img):
    model = DenseNet201()
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    filename = 'temp.jpg'
    img = Image.fromarray(img)
    img.save(filename)
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img,axis=0)
    feature = fe.predict(img, verbose=0)
    #image_id = 'abc'
    #feature[image_id] = feature
    return feature

def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

# Define the prediction function
def caption_predictor(model, tokenizer, max_length, feature):

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = caption_model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None or word == 'endseq':
            break
        
        in_text += " " + word
        
    return in_text

def predict(image):
    feature = extract_feature(image)
    description = caption_predictor(caption_model, tokenizer, max_length, feature)
    description = description.replace('startseq ', '').replace('endseq', '')
    print(description)
    return description

# Define Gradio interface
inputs = gr.Image(label="Input Image")
outputs = gr.Textbox(label="Generated Caption")
iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Image Captioning")

# Launch the interface
iface.launch(share=True)
