from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import save_model

# Load the pretrained model (without the top layer for customization)
model = ResNet50V2(weights="imagenet")

# Save the model as an .h5 file
model.save("ResNet50V2.h5")

print("Model downloaded and saved as ResNet50V2.h5")