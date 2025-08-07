#-------object tracking-----------

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO

# 🚀 Setup Streamlit UI
st.set_page_config(page_title="Data Collection India", layout="wide")
st.title("Data Collection India | Interview Assessment")

# 📁 Sidebar config
with st.sidebar:
    st.header("Upload & Configure")
    model_file = "./weight/best.pt"
    data_type = st.radio("Choose Input Type", ["Image", "Video"])
    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    # iou_thresh = st.slider("IoU Threshold (Video Tracking)", 0.1, 1.0, 0.5, 0.05)
    iou_thresh = st.slider("IoU Threshold (Video Tracking)", 0.1, 1.0, 0.5, 0.05)
    tracker_type = st.selectbox("Select Tracker Type", ["bytetrack", "botsort"], index=0)

    input_file = st.file_uploader(
        f"Upload {'Image' if data_type == 'Image' else 'Video'}",
        type=["jpg", "jpeg", "png"] if data_type == "Image" else ["mp4", "avi"]
    )

# 🧐 Load model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_file)

# 📄 Toggle training report viewer
if "show_report" not in st.session_state:
    st.session_state.show_report = False

if st.button("📄 View Model Training Report"):
    st.session_state.show_report = not st.session_state.show_report

report_text = """
### **YOLOv11 Object Detection - Model Training Report**

#### **1. Code Flow / Algorithm**
1. Load YOLOv11m pretrained model (`yolo11m.pt`)
2. Load and parse dataset defined in `data.yaml`
3. Set training configurations: `epochs=50`, `batch size=8`, `device=GPU`
4. Train the model on 5048 images
5. Validate the model on 561 images
6. Evaluate model metrics and save best-performing weights

#### **2. Exploratory Data Analysis (EDA)**
**Dataset Summary:**  
- Training Images: 5048  
- Validation/Test Images: 561  

**Classes:**  
- ADVISORY_SPEED_MPH: 188 instances  
- DIRECTIONAL_ARROW_AUXILIARY: 199 instances  
- DO_NOT_ENTER: 220 instances

#### **3. Training Configuration**
- Model: YOLOv11m  
- Epochs: 50  
- Batch Size: 8  
- Patience: 0  
- Device: GPU (Google Colab)

#### **Why Only 50 Epochs?**
Google Colab’s time and resource limits required training to stop at 50 epochs. Despite this, the model achieved high accuracy and generalization.

#### **4. Training Results Analysis**
- **mAP@0.5:** 0.979  
- **mAP@0.5:0.95:** 0.871  
- **Precision:** 0.947  
- **Recall:** 0.959  

**Class-wise Performance:**

| Class                         | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------------------------------|-----------|--------|---------|---------------|
| ADVISORY_SPEED_MPH           | 0.975     | 0.995  | 0.994   | 0.915         |
| DIRECTIONAL_ARROW_AUXILIARY  | 0.930     | 0.905  | 0.958   | 0.775         |
| DO_NOT_ENTER                 | 0.936     | 0.977  | 0.986   | 0.923         |

**Inference Speed:**  
- Preprocess: 0.3 ms/image  
- Inference: 7.5 ms/image  
- Postprocess: 3.1 ms/image

#### **5. Confusion Matrix Analysis**
- ADVISORY_SPEED_MPH: 187 correct, 9 missed  
- DIRECTIONAL_ARROW_AUXILIARY: 188 correct, 29 missed  
- DO_NOT_ENTER: 218 correct, 31 missed  
- 14 false positives from background

#### **6. Detection on Unseen Data**
Performs reliably across various sizes and lighting. Cluttered background is the major error source.

#### **Conclusion:**
High precision, low latency YOLOv11m model suitable for real-time object detection.
"""

if st.session_state.show_report:
    with st.expander("Model Training Report", expanded=True):
        st.markdown(report_text)
        try:
            with open("./Report/Training_Report_DCIL.pdf", "rb") as file:
                st.download_button(
                    label="📅 Download Report as PDF",
                    data=file.read(),
                    file_name="Training_Report_DCIL.pdf",
                    mime="application/pdf"
                )
        except FileNotFoundError:
            st.warning("⚠️ Report PDF not found in ./Report directory.")

# 🔍 Inference preview
st.subheader("🔍 Model Inference Preview")

if input_file is not None:
    if data_type == "Image":
        img = Image.open(input_file).convert("RGB")
        img_np = np.array(img)

        results = model.predict(img_np, conf=conf_thresh)
        annotated_img = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_np, caption="📸 Original Image", use_column_width=True)
        with col2:
            st.image(annotated_img, caption="🎯 Detected Image", use_column_width=True)

    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file.read())
        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)
        stframe1, stframe2 = st.columns(2)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # results = model.track(frame, conf=conf_thresh, iou=iou_thresh, persist=True)
            results = model.track(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                tracker=tracker_type,
                persist=True
            )
            annotated_frame = results[0].plot()

            with stframe1:
                st.image(rgb_frame, caption="🎞️ Original Frame", use_column_width=True)
            with stframe2:
                st.image(annotated_frame, caption="🔍 Tracked Output", use_column_width=True)

        cap.release()
else:
    st.warning("Please upload an input file to begin inference.")



# ---------------Deployed working------------

# import streamlit as st
# from PIL import Image
# import numpy as np
# import cv2
# import tempfile
# from ultralytics import YOLO

# # 🚀 Setup Streamlit UI
# st.set_page_config(page_title="Data Collection India", layout="wide")
# st.title("Data Collection India | Interview Assessment")

# # 📁 Sidebar config
# with st.sidebar:
#     st.header("Upload & Configure")
#     model_file = "./weight/best.pt"
#     data_type = st.radio("Choose Input Type", ["Image", "Video"])
#     conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
#     iou_thresh = st.slider("IoU Threshold (Video Tracking)", 0.1, 1.0, 0.5, 0.05)

#     input_file = st.file_uploader(
#         f"Upload {'Image' if data_type == 'Image' else 'Video'}",
#         type=["jpg", "jpeg", "png"] if data_type == "Image" else ["mp4", "avi"]
#     )

# # 🧠 Load model
# @st.cache_resource
# def load_model(path):
#     return YOLO(path)

# model = load_model(model_file)

# # 📄 Toggle training report viewer
# if "show_report" not in st.session_state:
#     st.session_state.show_report = False

# if st.button("📄 View Model Training Report"):
#     st.session_state.show_report = not st.session_state.show_report

# report_text = """
# ### **YOLOv11 Object Detection - Model Training Report**

# #### **1. Code Flow / Algorithm**
# 1. Load YOLOv11m pretrained model (`yolo11m.pt`)
# 2. Load and parse dataset defined in `data.yaml`
# 3. Set training configurations: `epochs=50`, `batch size=8`, `device=GPU`
# 4. Train the model on 5048 images
# 5. Validate the model on 561 images
# 6. Evaluate model metrics and save best-performing weights

# #### **2. Exploratory Data Analysis (EDA)**
# **Dataset Summary:**  
# - Training Images: 5048  
# - Validation/Test Images: 561  

# **Classes:**  
# - ADVISORY_SPEED_MPH: 188 instances  
# - DIRECTIONAL_ARROW_AUXILIARY: 199 instances  
# - DO_NOT_ENTER: 220 instances

# #### **3. Training Configuration**
# - Model: YOLOv11m  
# - Epochs: 50  
# - Batch Size: 8  
# - Patience: 0  
# - Device: GPU (Google Colab)

# #### **Why Only 50 Epochs?**
# Google Colab’s time and resource limits required training to stop at 50 epochs. Despite this, the model achieved high accuracy and generalization.

# #### **4. Training Results Analysis**
# - **mAP@0.5:** 0.979  
# - **mAP@0.5:0.95:** 0.871  
# - **Precision:** 0.947  
# - **Recall:** 0.959  

# **Class-wise Performance:**

# | Class                         | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
# |------------------------------|-----------|--------|---------|---------------|
# | ADVISORY_SPEED_MPH           | 0.975     | 0.995  | 0.994   | 0.915         |
# | DIRECTIONAL_ARROW_AUXILIARY  | 0.930     | 0.905  | 0.958   | 0.775         |
# | DO_NOT_ENTER                 | 0.936     | 0.977  | 0.986   | 0.923         |

# **Inference Speed:**  
# - Preprocess: 0.3 ms/image  
# - Inference: 7.5 ms/image  
# - Postprocess: 3.1 ms/image

# #### **5. Confusion Matrix Analysis**
# - ADVISORY_SPEED_MPH: 187 correct, 9 missed  
# - DIRECTIONAL_ARROW_AUXILIARY: 188 correct, 29 missed  
# - DO_NOT_ENTER: 218 correct, 31 missed  
# - 14 false positives from background

# #### **6. Detection on Unseen Data**
# Performs reliably across various sizes and lighting. Cluttered background is the major error source.

# #### **Conclusion:**
# High precision, low latency YOLOv11m model suitable for real-time object detection.
# """

# if st.session_state.show_report:
#     with st.expander("Model Training Report", expanded=True):
#         st.markdown(report_text)
#         try:
#             with open("./Report/Training_Report_DCIL.pdf", "rb") as file:
#                 st.download_button(
#                     label="📥 Download Report as PDF",
#                     data=file.read(),
#                     file_name="Training_Report_DCIL.pdf",
#                     mime="application/pdf"
#                 )
#         except FileNotFoundError:
#             st.warning("⚠️ Report PDF not found in ./Report directory.")

# # 🔍 Inference preview
# st.subheader("🔍 Model Inference Preview")

# if input_file is not None:
#     if data_type == "Image":
#         img = Image.open(input_file).convert("RGB")
#         img_np = np.array(img)

#         results = model.predict(img_np, conf=conf_thresh)
#         annotated_img = results[0].plot()

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(img_np, caption="📸 Original Image", use_column_width=True)
#         with col2:
#             st.image(annotated_img, caption="🎯 Detected Image", use_column_width=True)

#     else:
#         temp_file = tempfile.NamedTemporaryFile(delete=False)
#         temp_file.write(input_file.read())
#         video_path = temp_file.name

#         cap = cv2.VideoCapture(video_path)
#         stframe1, stframe2 = st.columns(2)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = model.track(frame, conf=conf_thresh, iou=iou_thresh, persist=True) 
#             annotated_frame = results[0].plot()

#             with stframe1:
#                 st.image(rgb_frame, caption="🎞️ Original Frame", use_column_width=True)
#             with stframe2:
#                 st.image(annotated_frame, caption="🔍 Tracked Output", use_column_width=True)

#         cap.release()
# else:
#     st.warning("Please upload an input file to begin inference.")
