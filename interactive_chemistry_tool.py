import cv2
import numpy as np
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import os
from dotenv import load_dotenv



# Load environment variables for API keys
load_dotenv()

# Initialize Text-to-Speech Engin
tts_engine = pyttsx3.init()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Gesture-to-Element Mapping


# Gesture-to-Element Mapping
gesture_to_element = {
    "one_finger": "Oxygen (O)",
    "two_fingers": "Hydrogen (H)"
}

# Selected Elements
selected_elements = []

# Function for Text-to-Speech
def narrate(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to Count Fingers Using Convexity Defects
def count_fingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        start, end, far, _ = defects[i, 0]
        start = tuple(contour[start][0])
        end = tuple(contour[end][0])
        far = tuple(contour[far][0])

        # Distance calculation
        a = np.linalg.norm(np.array(start) - np.array(far))
        b = np.linalg.norm(np.array(end) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

        # If the angle is less than 90 degrees, it's a valid finger
        if angle <= np.pi / 2:
            finger_count += 1

    return finger_count + 1  # Add thumb

def detect_gesture(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological transformations
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 3000:  # Reduced minimum area
            return count_fingers(max_contour)

    return None

# LangChain Setup
prompt_template = """
Provide a detailed explanation about the following chemical element or compound: {text}.
Include its uses, properties, and significance.
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm_chain = prompt | llm

# Function to Generate Explanations Using LangChain
def generate_explanation(text):
    try:
        explanation = llm_chain.invoke({"text": text})
        return explanation
    except Exception as e:
        return f"Error generating explanation: {e}"

# Function to Update Tkinter GUI
def update_gui(elements, explanation=None):
    for widget in frame.winfo_children():
        widget.destroy()

    label = tk.Label(frame, text="Selected Elements:", font=("Arial", 14))
    label.pack()

    for element in elements:
        element_label = tk.Label(frame, text=f"- {element}", font=("Arial", 12))
        element_label.pack()

    if explanation:
        explanation_label = tk.Label(frame, text=f"Explanation:\n{explanation}", font=("Arial", 12), fg="blue", wraplength=400, justify="left")
        explanation_label.pack()

# Create Tkinter Window
window = tk.Tk()
window.title("Interactive Chemistry Tool")
window.geometry("600x700")

frame = tk.Frame(window)
frame.pack(pady=20)

label_img = tk.Label(window)
label_img.pack()

cap = cv2.VideoCapture(0)

while True:
    success, frame_cv = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    frame_cv = cv2.flip(frame_cv, 1)
    gesture = detect_gesture(frame_cv)

    if gesture in gesture_to_element:
        element = gesture_to_element[gesture]
        if element not in selected_elements:
            selected_elements.append(element)
            narrate(f"{element} selected")

    compound = None
    if "Oxygen (O)" in selected_elements and "Hydrogen (H)" in selected_elements:
        compound = "H2O"
        selected_elements.clear()

    explanation = None
    if compound:
        explanation = generate_explanation(compound)
    elif selected_elements:
        explanation = generate_explanation(" and ".join(selected_elements))

    update_gui(selected_elements, explanation)

    img = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_pil)

    label_img.config(image=img_tk)
    label_img.image = img_tk

    window.update_idletasks()
    window.update()

cap.release()
cv2.destroyAllWindows()
