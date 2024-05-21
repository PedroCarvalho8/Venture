import cv2
import streamlit as st
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

col1, col2 = st.columns(2)

if col1.button("Iniciar"):
    if col2.button("Parar"):
        st.rerun()

    with st.empty():
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(frame)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                xyxys = boxes.xyxy
                
                names = result.names
                cls = int(result.boxes.cls[0])

                for xyxy in xyxys:
                    
                    cv2.putText(frame, names[cls], (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3)


            st.image(frame)
           
    

