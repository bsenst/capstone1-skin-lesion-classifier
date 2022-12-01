import numpy as np
import keras
import huggingface_hub
from huggingface_hub import Repository
from tensorflow.keras.preprocessing.image import load_img
from transformers import ImageClassificationPipeline
from transformers import TFAutoModel
from transformers import TFPreTrainedModel
from tensorflow.keras.applications.xception import preprocess_input

repo = Repository(local_dir="repo", clone_from="bsenst/skin-cancer-HAM10k")
model = keras.models.load_model("repo/xception_v4_1_07_0.699.h5")

# img = load_img(input("Enter filepath of an skin lesion image (i.e. Melanoma.jpg): "), target_size=(150, 150))
img = load_img("skinlesion.jfif", target_size=(150, 150))

x = np.array(img)
X = np.array([x])
X.shape

X = preprocess_input(X)
pred = model.predict(X)

classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

def image_classifier(inp):
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)
    pred = model.predict(X).flatten()
    return {classes[i]: float(pred[i]) for i in range(7)}

print(image_classifier(img))
