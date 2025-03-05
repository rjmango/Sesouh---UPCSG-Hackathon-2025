# MedVision by Sesuoh - Medical Waste Object Detection

MedVision by Sesuoh is an advanced object detection system for identifying and classifying medical waste. The project leverages the YOLOv8 model by Ultralytics and is hosted using Streamlit for an interactive user experience. The model has been trained on over a thousand images to ensure accurate detection and categorization of medical waste. Users can upload images or videos, adjust confidence levels, and filter specific classes for detection. Additionally, the system provides disposal recommendations based on the identified waste type.

## Features
- **Input Options:** Accepts both images and video files.
- **Real-Time Prediction:** Detects and classifies medical waste items.
- **Confidence Adjustment:** Allows users to modify confidence levels for prediction.
- **Class Filtering:** Users can include or exclude specific waste categories.
- **Disposal Suggestions:** Provides proper waste disposal guidance based on detected objects.

## Model Training
- **Model Used:** YOLOv8 by Ultralytics
- **Dataset:** Trained on a dataset of over 1,000 images containing various medical waste objects.
- **Evaluation Metrics:**
  - Mean Average Precision (mAP@0.5): **0.717**
  - Recall: **0.87** at **0.000 Confidence**
  - Precision-Confidence Analysis: Demonstrates stable precision with increasing confidence levels.
  - Precision-Recall Curve: Illustrates balanced trade-offs between precision and recall for different waste categories.
  
### F1 Scores Per Class
| Class         | F1 Score |
|--------------|---------|
| IV Tube      | 0.716   |
| Bandage      | 0.938   |
| Cotton       | 0.672   |
| Gloves       | 0.673   |
| Mask         | 0.734   |
| Medical Cap  | 0.891   |
| Needle       | 0.499   |
| Scissors     | 0.547   |
| Syringe      | 0.841   |
| Test Tube    | 0.716   |
| Vial         | 0.829   |
| Waste        | 0.544   |

- **Mean F1 Score:** **0.717**

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/rjmango/Sesouh---UPCSG-Hackathon-2025.git
   cd Sesouh---UPCSG-Hackathon-2025
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload an image or video file.
2. Adjust the confidence threshold in the sidebar.
3. Select or exclude specific classes for detection.
4. Click "Predict" to run the YOLOv8 model and obtain results.
5. Review detected objects along with disposal suggestions.

## Acknowledgments
- **Ultralytics:** For providing the powerful YOLOv8 model for object detection.
- **Streamlit:** For making interactive deployment simple and effective.
- **Our Team, Sesuoh:** For dedicated research and development in medical waste detection.

## Future Improvements
- Enhance dataset diversity for better accuracy.
- Implement real-time video streaming analysis.
- Integrate AI model into various IoT solutions for flexibility and further use cases.
- Develop a mobile-friendly version for wider accessibility.

---
**MedVision by Sesuoh - We find waste.**

