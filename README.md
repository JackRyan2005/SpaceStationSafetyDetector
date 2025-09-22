# SpaceStationSafetyDetectorusingYOLOV11(Using ROBOFLOW)
🚀 Space Station Safety Object Detection
========================================

This project focuses on detecting safety-critical objects in a simulated space station environment using a YOLOv11 object detection model. The primary goal is to accurately identify, visualize, and count essential items like fire extinguishers, oxygen tanks, and emergency phones to enhance safety monitoring protocols in real-world aerospace applications.

📊 Dataset
----------

The model was trained on a curated dataset of images from a simulated space station environment.

*   **Source**: Falcon Simulator Dataset (Space Station Safety Objects)
    
*   **Total Images**: ~10,300 (including augmentations)
    
*   **Data Split**:
    
    *   Training: ~70%
        
    *   Validation: ~20%
        
    *   Testing: ~10%
        
*   **Augmentations Applied**: To improve model robustness, the following augmentations were used:
    
    *   Horizontal Flip
        
    *   Brightness & Exposure Adjustments
        
    *   Rotation (±15° to ±30°)
        

### Classes (7)

The dataset includes the following object classes:

**Class NameCount**Oxygen Tank1,457Nitrogen Tank1,400First Aid Box841Fire Extinguisher793Fire Alarm264Safety Switch Panel255Emergency Phone238

🔹 Model
--------

*   **Architecture**: YOLOv11 (latest Ultralytics version)
    
*   **Framework**: Roboflow (trained with a COCO backbone)
    
*   **Task**: Object Detection
    

### Workflow (Detect–Visualize–Count)

The model follows a three-step process:

1.  **Detection**: Identifies objects and generates bounding boxes, class labels, and confidence scores.
    
2.  **Visualization**: Overlays bounding boxes and labels onto the input images for clear interpretation.
    
3.  **Counting**: Aggregates the detected objects to provide a total count for each class.
    

📊 Model Performance
--------------------

The model's performance was evaluated on the validation set, achieving the following metrics:

**MetricScoremAP@5076.3%Precision82.2%Recall70.9%**

### Training Graphs

Below are the training and validation performance graphs, showing metrics such as loss and mean Average Precision (mAP) over epochs.

#### Model Performance

#### Advanced Training Graphs

🛰️ Why This Matters
--------------------

*   **Robustness**: The use of data augmentation makes the model more resilient to variations in lighting, rotation, and perspective.
    
*   **High Accuracy**: YOLOv11 provides a strong backbone for object detection, ensuring efficient and accurate inference.
    
*   **Practical Application**: Beyond simple detection, the counting feature makes this model a valuable tool for safety and compliance monitoring in critical environments like space stations.
    

✅ Conclusion
------------

With a **mAP@50 of 76.3%**, strong precision (~82%), and reliable recall (~71%), this model demonstrates its capability as an effective safety monitoring tool. It can be integrated into automated systems to ensure that all necessary safety equipment is present and accounted for in a space station or other high-stakes environments.
