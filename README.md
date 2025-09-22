
# 🚀 Space Station Safety Object Detection using YOLOv11 (with Roboflow)

This project focuses on detecting safety-critical objects in a simulated space station environment using a **YOLOv11 object detection model**. The primary goal is to accurately identify, visualize, and count essential items like fire extinguishers, oxygen tanks, and emergency phones to enhance safety monitoring protocols in real-world aerospace applications.

---

## 🛠️ Initial Setup and Workflow

The project was developed following these key steps:

1.  **Data Collection**: The initial dataset was sourced from the Falcon Simulator, providing images of a simulated space station environment.
2.  **Data Upload to Roboflow**: The collected images were uploaded to the Roboflow platform, which served as the primary tool for dataset management and model training.
3.  **Data Preprocessing & Augmentation**: Within Roboflow, the dataset was cleaned and prepared. Augmentations such as flips, brightness adjustments, and rotations were applied to increase the dataset's diversity and improve model robustness.
4.  **Model Training**: The preprocessed dataset was used to train a YOLOv11 object detection model directly on the Roboflow platform, leveraging its streamlined training infrastructure with a COCO backbone.
5.  **Model Evaluation**: After training, the model's performance was evaluated using standard object detection metrics (mAP, Precision, and Recall) provided by Roboflow's built-in analysis tools.
6.  **Inference and Application**: The trained model was used for inference. A simple application was built using **Streamlit in a Google Colab environment** to run the model on new images and visualize the detections.

---

## 📊 Dataset

The model was trained on a curated dataset of images from a simulated space station environment.

*   **Source**: Falcon Simulator Dataset (Space Station Safety Objects)
*   **Total Images**: ~10,300 (after augmentations)
*   **Data Split**:
    *   **Training**: ~70%
    *   **Validation**: ~20%
    *   **Testing**: ~10%
*   **Augmentations Applied**: Horizontal Flip, Brightness & Exposure Adjustments, Rotation (±15° to ±30°).

### Classes (7)

The dataset includes the following object classes with their respective counts:

| Class Name | Count |
| :--- | :--- |
| Oxygen Tank | 1,457 |
| Nitrogen Tank | 1,400 |
| First Aid Box | 841 |
| Fire Extinguisher | 793 |
| Fire Alarm | 264 |
| Safety Switch Panel | 255 |
| Emergency Phone | 238 |

![Class Labels](https://i.postimg.cc/q7mJC3nC/Screenshot-2025-09-22-164956.png)

---

## 🔹 Model

*   **Architecture**: YOLOv11 (latest Ultracyclics version)
*   **Framework**: Roboflow (trained with a COCO backbone)
*   **Task**: Object Detection

### Workflow (Detect–Visualize–Count)

The model follows a three-step process:

1.  **Detection**: Identifies objects and generates bounding boxes, class labels, and confidence scores.
2.  **Visualization**: Overlays bounding boxes and labels onto the input images for clear interpretation.
3.  **Counting**: Aggregates the detected objects to provide a total count for each class.

![Model Workflow](https://i.postimg.cc/Y0H1GZQ9/Screenshot-2025-09-22-171715.png)

---

## 📊 Model Performance

The model's performance was evaluated on the validation set, achieving the following metrics:

| Metric | Score |
| :--- | :--- |
| **mAP@50** | **76.3%** |
| **Precision** | **82.2%** |
| **Recall** | **70.9%** |

![Metrics](https://i.postimg.cc/JhKw2Czw/Screenshot-2025-09-22-164830.png)

### Training Graphs

Below are the training and validation performance graphs, showing metrics such as loss and mean Average Precision (mAP) over epochs.

**Model Performance**![Model Performance 1](https://i.postimg.cc/7hnSg9NN/Screenshot-2025-09-22-171055.png)![Model Performance 2](https://i.postimg.cc/rFq1868Z/Screenshot-2025-09-22-171110.png)

**Advanced Training Graphs**
![Results 1](https://i.postimg.cc/FHR0Ljkj/Screenshot-2025-09-22-171027.png)

---

## 🚀 Interactive Demo Application

To provide a hands-on demonstration of the model's capabilities, an interactive web application was developed using **Streamlit**. The application is designed to be run in a **Google Colab** notebook and exposed to the public internet using **ngrok**. This setup allows anyone to upload an image and see the model's detections in real-time without any local installation.

### How to Run the Demo

You can launch the interactive demo by running the following Python code in a Google Colab notebook. Make sure to replace the placeholder API keys with your own.

```python
# =========================================================
# 🚀 Streamlit + Roboflow YOLOv11 Inference + ngrok in Colab
# =========================================================

# Step 1: Install dependencies
!pip install -q streamlit pyngrok inference-sdk opencv-python-headless pillow

# Step 2: Define Streamlit App
app_code = """
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os, io, cv2
import numpy as np
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="YOLOv11 Safety Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown('''
<style>
    .stApp { background: linear-gradient(120deg, #eef2f7, #dde7f2); }
    h1 { color: #2c3e50; text-align: center; }
    .block-container { padding-top: 1rem; }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #34495e;
        color: #f1f1f1;
    }
</style>
''', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ Safety Equipment Detector")
    st.info("Upload an image and let the YOLOv11 model (Roboflow) detect safety equipment.")

    st.markdown("### 📊 Model Performance")
    st.metric("mAP@50", "76.3%")
    st.metric("Precision", "82.2%")
    st.metric("Recall", "70.9%")

    st.markdown("### 📦 Dataset Classes")
    dataset_counts = {
        "Emergency Phone": 238,
        "Fire Alarm": 264,
        "Fire Extinguisher": 793,
        "First Aid Box": 841,
        "Nitrogen Tank": 1400,
        "Oxygen Tank": 1457,
        "Safety Switch Panel": 255
    }
    st.table(pd.DataFrame(list(dataset_counts.items()), columns=["Class", "Count"]))

# --- Main Title ---
st.title("🚀 YOLOv11 Safety Equipment Detection")

# --- Initialize Roboflow Client ---
API_KEY = "YOUR_ROBOFLOW_API_KEY"  # 🔑 Replace with your Roboflow API key
WORKSPACE_NAME = "xwork-bjgiu"
WORKFLOW_ID = "detect-count-and-visualize-5"

try:
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )
except Exception as e:
    st.error(f"Error initializing client: {e}")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Save temporary image
    image_bytes = uploaded_file.getvalue()
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_bytes)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Uploaded Image")
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Original", use_column_width=True)

    with col2:
        st.subheader("🔍 Detection Results")
        with st.spinner("Running YOLOv11 inference..."):
            try:
                result = client.run_workflow(
                    workspace_name=WORKSPACE_NAME,
                    workflow_id=WORKFLOW_ID,
                    images={"image": temp_image_path},
                    use_cache=True
                )

                # ✅ Handle visualization if provided
                if result and isinstance(result, list):
                    res = result[0]

                    if "visualization" in res and res["visualization"]:
                        st.image(res["visualization"], caption="Detection Visualization", use_column_width=True)

                    # ✅ Handle predictions
                    predictions = res.get("predictions", {}).get("predictions", [])
                    if predictions:
                        # Build dataframe
                        rows = []
                        class_counts = {}
                        for pred in predictions:
                            cname = pred.get("class", "Unknown")
                            conf = round(pred.get("confidence", 0)*100, 2)
                            class_counts[cname] = class_counts.get(cname, 0) + 1
                            rows.append([cname, conf, pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")])

                        st.markdown("### 📊 Detected Objects")
                        st.table(pd.DataFrame(rows, columns=["Class", "Confidence %", "X", "Y", "Width", "Height"]))

                        st.markdown("### 🔢 Object Counts")
                        st.table(pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]))

                    else:
                        st.warning("⚠️ No objects detected in this image.")

                    with st.expander("Show Raw JSON Output"):
                        st.json(result)
                else:
                    st.error("Unexpected model response")
                    st.json(result)

            except Exception as e:
                st.error(f"Inference failed: {e}")

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
"""

# Step 3: Write app.py
with open("app.py", "w") as f:
    f.write(app_code)

# Step 4: ngrok setup
from pyngrok import ngrok
ngrok.kill()
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"   # 🔑 Replace with your ngrok token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8501)
print("="*60)
print("🚀 Streamlit App is LIVE")
print("🔗 Public URL:", public_url)
print("="*60)

# Step 5: Run Streamlit (non-blocking)
!nohup streamlit run app.py --server.port 8501 &
```

---

## 🖥️ Final Application Interface

The Streamlit application provides a clean and user-friendly interface for interacting with the model. Users can upload an image and view the original image alongside the annotated output, with detailed tables for detected objects and their counts.

[![Final Application Screenshot](https://i.postimg.cc/rwXVnKsK/Screenshot-2025-09-22-183635.png)](https://postimg.cc/YhRBj21H)

---

## ✅ Conclusion

With a **mAP@50 of 76.3%**, strong precision (~82%), and reliable recall (~71%), this model demonstrates its capability as an effective safety monitoring tool. It can be integrated into automated systems to ensure that all necessary safety equipment is present and accounted for in a space station or other high-stakes environments.

---

## 🔮 Future Enhancements

While the current model performs well, there are several avenues for future improvement:

*   **Expand the Dataset**: Incorporate a wider variety of images, including different lighting conditions, camera angles, and object occlusion to improve generalization.
*   **Train on Real-World Imagery**: While the simulated dataset is a great starting point, training on real images from space station modules would significantly enhance the model's practical utility.
*   **Hyperparameter Tuning**: Fine-tune model parameters such as learning rate, batch size, and optimizer settings to potentially boost performance metrics.
*   **Real-Time Alert System**: Integrate the model with a live video feed to create a real-time monitoring system that can trigger alerts if a critical safety item is missing or obstructed.
*   **Edge Deployment**: Optimize the model for deployment on edge devices (like a Raspberry Pi or NVIDIA Jetson) for low-latency, on-site processing without relying on a constant cloud connection.

---

## 👥 Team Members

*   V C Premchand Yadav
*   Edupulapati Sai Praneeth
*   P R Kiran Kumar Reddy
*   K Sri Harsha Vardhan
*   Liel Stephen
*   Suheb Nawab Sheikh
---

---

##  Thank You  🙏 

Special thanks to the following organizations that made this project possible:

- **[Duality AI](https://www.linkedin.com/company/dualityai/)**  
  [![Duality AI](https://img.shields.io/badge/Duality%20AI-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/dualityai/)  
  Thank you for this Duality AI  

- **[Roboflow](https://roboflow.com/)**  
  [![Roboflow](https://img.shields.io/badge/Roboflow-8A2BE2?logo=roboflow&logoColor=white)](https://roboflow.com/)  
  Thanks to Roboflow for providing tools to annotate and train our data


