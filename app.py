# streamlit run app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO

model = YOLO("best.pt")

class_dict = {
    0: "IVtube",
    1: "bandage",
    2: "cotton",
    3: "gloves",
    4: "mask",
    5: "medical cap",
    6: "needle",
    7: "scissors",
    8: "syringe",
    9: "test tube",
    10: "vial",
    11: "waste",
}

all_class_names = list(class_dict.values())


np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(all_class_names), 3), dtype="uint8")


def parse_yolo_results(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    detections = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        detections.append([x1, y1, x2, y2, conf[i], int(cls[i])])
    return detections


def draw_boxes(image, detections, selected_classes):

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    for x1, y1, x2, y2, conf, cls_id in detections:
        class_name = all_class_names[cls_id]
        if class_name not in selected_classes:
            continue

        color = [int(c) for c in colors[cls_id]]
        label = f"{class_name} {conf:.2f}"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return image


def detect_image(uploaded_image, selected_classes, conf_thres):

    pil_img = Image.open(uploaded_image).convert("RGB")
    results = model.predict(source=pil_img, conf=conf_thres)
    detections = parse_yolo_results(results[0])
    annotated_img = draw_boxes(pil_img, detections, selected_classes)
    return annotated_img


def detect_video(video_file, selected_classes, conf_thres):

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=frame_rgb, conf=conf_thres, verbose=False)
        detections = parse_yolo_results(results[0])
        annotated_frame = draw_boxes(frame_rgb, detections, selected_classes)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()


def detect_webcam(selected_classes, conf_thres):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No camera found!")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=frame_rgb, conf=conf_thres, verbose=False)
        detections = parse_yolo_results(results[0])
        annotated_frame = draw_boxes(frame_rgb, detections, selected_classes)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


def main():
    st.set_page_config(page_title="MedVision", page_icon="icon.ico")
    st.logo("logo.svg")
    st.title("We find waste.")
    # st.image("logo.svg", use_container_width=True)

    with st.sidebar:
        st.header("Configure model settings")

        conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

        selected_classes = st.multiselect(
            "Select Classes", options=all_class_names, default=all_class_names
        )

    mode = st.selectbox(
        "Select Activity",
        ["Detect from Image", "Detect from Video", "Detect from Webcam"],
    )

    if mode == "Detect from Image":
        uploaded_file = st.file_uploader(
            "Upload an Image", type=["jpg", "jpeg", "png"], key="image_uploader"
        )
        detect_image_button = st.button("Detect Image")

        if uploaded_file is not None and detect_image_button:
            annotated_img = detect_image(uploaded_file, selected_classes, conf_thres)

            if annotated_img is not None:
                st.success("Image processed successfully 🎉")
                st.image(
                    annotated_img, caption="Detection Results", use_container_width=True
                )

    elif mode == "Detect from Video":
        uploaded_video = st.file_uploader(
            "Upload a Video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )
        detect_video_button = st.button("Detect Video")

        if uploaded_video is not None and detect_video_button:
            detect_video(uploaded_video, selected_classes, conf_thres)

    else:
        detect_webcam_button = st.button("Start Webcam Detection")

        if "detect_webcam_button" in locals() and detect_webcam_button:
            detect_webcam(selected_classes, conf_thres)

    # if mode == "Detect from Image":
    #     if uploaded_file is not None and detect_image_button:
    #         annotated_img = detect_image(uploaded_file, selected_classes, conf_thres)

    #         if annotated_img is not None:
    #             st.success("Image processed successfully 🎉")
    #             st.image(
    #                 annotated_img, caption="Detection Results", use_container_width=True
    #             )

    # elif mode == "Detect from Video":
    #     st.text("Processing video...")
    #     if uploaded_video is not None and detect_video_button:
    #         detect_video(uploaded_video, selected_classes, conf_thres)

    # else:
    #     if "detect_webcam_button" in locals() and detect_webcam_button:
    #         detect_webcam(selected_classes, conf_thres)


if __name__ == "__main__":
    main()
