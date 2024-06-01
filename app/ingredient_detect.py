import streamlit as st
import cv2
import numpy as np
from PIL import Image
import openai
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from transformers import pipeline
sys.path.append("D:\\Documents\\Studies\\MS\\UB\\MY Masters\\SPR SEM 2024\\Deep Learning\\Project\\APP\\yolov7")  # Adjust this to the path where you cloned YOLOv7

from models.experimental import attempt_load 

model_path ="D:\\Documents\\Studies\\MS\\UB\\MY Masters\\SPR SEM 2024\\Deep Learning\\Project\\APP\\best.pt"
# add your oprn api k here
def load_model(model_path):
    """Load the YOLOv7 trained model."""
    device = torch.device('cpu') 
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model

def detect_ingredients(image, model):
    """Use the YOLOv7 model to detect ingredients in the image."""
    image = np.array(image)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        predictions = outputs[0]
        print("Predictions Shape:", predictions.shape)
    return predictions


def get_recipe_from_gpt(ingredients):
    prompt = f"Create a recipe using the following ingredients: {', '.join(ingredients)}"
    generator = pipeline("text-generation", model="gpt2")
    recipe = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].strip()
    
    return recipe


def parse_predictions(predictions, class_labels, threshold=0.5):
    """Parse model output to extract detected class labels with a confidence threshold."""
    predictions = predictions.squeeze(0)  
    detected_ingredients = []

    for prediction in predictions:
        conf = prediction[4]  
        if conf > threshold:
            class_probs = prediction[5:]  
            class_id = class_probs.argmax()  
            class_confidence = class_probs[class_id]
            if class_confidence > threshold:
                ingredient = class_labels.get(int(class_id), "Unknown")
                if ingredient not in detected_ingredients:
                    detected_ingredients.append(ingredient)

    return detected_ingredients

def load_class_labels(filepath):
    class_labels = {}
    with open(filepath, 'r') as file:
        for idx, line in enumerate(file):
            class_labels[idx] = line.strip()
    return class_labels

def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("Eyes on Eats")
    st.header("Snap, Cook, Enjoy: Satisfy Your Cravings!")
    st.write("What's in Your Kitchen? Let's Create a Dish")
    st.write("MODEL: Object Detection using YOLO")

    model = load_model(model_path)
    class_labels = load_class_labels("D:\\Documents\\Studies\\MS\\UB\\MY Masters\\SPR SEM 2024\\Deep Learning\\Project\\APP\\Final_classes.txt")

    uploaded_file = st.file_uploader("Upload an image of the ingredients", type=["jpg", "jpeg", "png"])
    st.write("Uploaded Image")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            predictions = detect_ingredients(image, model)
            ingredients = parse_predictions(predictions, class_labels)

            st.write("Predicted Ingredients:", ingredients)
            
            recipe = get_recipe_from_gpt(ingredients)
            st.write("Generated Recipe:", recipe)

        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == '__main__':
    main()
