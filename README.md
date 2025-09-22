# 🚀 Space Station Safety Object Detection using YOLOv11 (with Roboflow)

This project focuses on detecting safety-critical objects in a simulated space station environment using a **YOLOv11 object detection model**. The primary goal is to accurately identify, visualize, and count essential items like fire extinguishers, oxygen tanks, and emergency phones to enhance safety monitoring protocols in real-world aerospace applications.

---

## 📊 Dataset

The model was trained on a curated dataset of images from a simulated space station environment.

*   **Source**: Falcon Simulator Dataset (Space Station Safety Objects)
*   **Total Images**: ~10,300 (after augmentations)
*   **Data Split**:
    *   **Training**: ~70%
    *   **Validation**: ~20%
    *   **Testing**: ~10%
*   **Augmentations Applied**: To improve model robustness, the following augmentations were used:
    *   Horizontal Flip
    *   Brightness & Exposure Adjustments
    *   Rotation (±15° to ±30°)

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

*   **Architecture**: YOLOv11 (latest Ultralytics version)
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

**Model Performance**
![Model Performance 1](https://i.postimg.cc/7hnSg9NN/Screenshot-2025-09-22-171055.png)![Model Performance 2](https://i.postimg.cc/rFq1868Z/Screenshot-2025-09-22-171110.png)

**Advanced Training Graphs**
![Results 1](https://i.postimg.cc/FHR0Ljkj/Screenshot-2025-09-22-171027.png)

---

## 🛰️ Why This Matters

*   **Robustness**: The use of data augmentation makes the model more resilient to variations in lighting, rotation, and perspective.
*   **High Accuracy**: YOLOv11 provides a strong backbone for object detection, ensuring efficient and accurate inference.
*   **Practical Application**: Beyond simple detection, the counting feature makes this model a valuable tool for safety and compliance monitoring in critical environments like space stations.

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
