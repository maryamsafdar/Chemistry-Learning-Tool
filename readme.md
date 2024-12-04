# ⚗️Chemistry Learning Tool

## Overview

The **⚗️Chemistry Learning Tool** allows users to create chemical compounds by using hand gestures, such as detecting extended fingers, and combines them into common compounds (e.g., water, sodium chloride). This application uses **Mediapipe** for hand gesture recognition, **OpenAI's GPT** (via LangChain) for generating detailed information about the compounds, and **Streamlit** for creating an interactive web interface.

## Features

- **Gesture Recognition**: Users can create compounds by performing specific hand gestures using a webcam.
- **Compound Formation**: Based on the hand gestures, the application combines elements into compounds (e.g., Water (H₂O), Sodium Chloride (NaCl)).
- **Real-time Feedback**: The app shows real-time updates of detected gestures, selected elements, and resulting compounds.
- **LLM Response**: After creating a compound, the app fetches and displays a detailed response from GPT about the compound.
- **Interactive Sidebar**: An interactive sidebar displaying selected gestures, elements, and compound information.

## Prerequisites

To run this project, you'll need to have the following dependencies installed:

- Python 3.x
- Streamlit
- OpenAI API (GPT)
- OpenCV
- Mediapipe

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/interactive-chemistry-tool.git
   cd interactive-chemistry-tool
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file in the project directory with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here

   ```

## How to Run

1. Run the Streamlit application:
   ```bash 
   streamlit run app.py
   ```

2. Open the application in your web browser (usually at http://localhost:8501).

## How It Works

The app uses **Mediapipe** to recognize hand gestures based on the number of extended fingers. Here's the mapping:

- 1 finger -> Hydrogen (H)
- 2 fingers -> Oxygen (O)
- 3 fingers -> Sodium (Na)
- 4 fingers -> Chlorine (Cl)
- 5 fingers -> Carbon (C)
These gestures are detected by tracking the landmarks on the user's hand.


## Compound Formation

When certain gestures are combined, the app forms chemical compounds. For example:

- Hydrogen (H) + Oxygen (O) = Water (H₂O)
- Sodium (Na) + Chlorine (Cl) = Sodium Chloride (NaCl)
- Carbon (C) + Oxygen (O) = Carbon Dioxide (CO₂)
The app then displays the formed compound in real-time.


## LLM Response
After a compound is formed, the app queries OpenAI's GPT (via LangChain) to provide more detailed information about the compound, which is displayed to the user.

### Sidebar and Interactive Feedback
The sidebar provides:

- Current gesture and element selected
- Compound formed based on the gestures
- Reset button to clear the selected elements and gestures

### Example Workflow
1. The user raises their hand in front of the webcam.
2. Based on the number of extended fingers, a corresponding element (e.g., Hydrogen, Oxygen) is detected.
3. The user combines elements to form a compound (e.g., Water).
3. The app queries OpenAI GPT to provide detailed information about the compound.
4. The information is displayed in the main window, and the sidebar is updated in real-time.

### Technologies Used

- **Streamlit:** For building the interactive web interface.
- **OpenCV:** For capturing the webcam feed.
- **Mediapipe:** For hand gesture recognition.
- **OpenAI GPT:** For generating detailed information about compounds.
- **LangChain:** For managing interactions with OpenAI GPT.
- **Python-dotenv:** For securely storing API keys.
