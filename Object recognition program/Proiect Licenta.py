import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Incarca modelul YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Incarca numele claselor
with open(r"C:\Users\debec\OneDrive\Desktop\Licenta YOLOv5\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def run_realtime_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converteste frame-ul la RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Efectueaza implementarea cu YOLOv5
        results = model(frame_rgb)

        # Extrage datele de detectie
        for idx in range(len(results.xyxy[0])):
            x1, y1, x2, y2, confidence, class_id = results.xyxy[0][idx].tolist()
            class_name = classes[int(class_id)]

            # Filtreaza detectiile cu incredere scazuta
            if confidence > 0.5:
                # Deseneaza o cutie de delimitare cu o culoare verde-albastruie
                blue_ish_green = (100, 255, 100)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), blue_ish_green, 2)

                # Adauga eticheta
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue_ish_green, 2)

        # Afiseaza iesirea
        cv2.imshow("Detectie Obiecte in Timp Real", frame)

        # Iesi la apasarea tastei 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Selecteaza Fisier Video", filetypes=[("Fisiere Video", "*.mp4;*.avi;*.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Converteste frame-ul la RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Efectueaza implementarea cu YOLOv5
            results = model(frame_rgb)

            # Extrage datele de detectie
            for idx in range(len(results.xyxy[0])):
                x1, y1, x2, y2, confidence, class_id = results.xyxy[0][idx].tolist()
                class_name = classes[int(class_id)]

                # Filtreaza detectiile cu incredere scazuta
                if confidence > 0.5:
                    # Deseneaza o cutie de delimitare cu o culoare verde-albastruie
                    blue_ish_green = (100, 255, 100)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), blue_ish_green, 2)

                    # Adauga eticheta
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue_ish_green, 2)

            # Afiseaza iesirea
            cv2.imshow("Detectie Obiecte in Timp Real", frame)

            # Iesi la apasarea tastei 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Detectie Obiecte")
    label = tk.Label(root, text="Selecteaza o optiune:")
    label.pack()
    button_camera = tk.Button(root, text="Rulare pe Camera", command=run_realtime_camera)
    button_camera.pack()
    button_video = tk.Button(root, text="Rulare pe Fisier Video", command=run_video_file)
    button_video.pack()
    root.mainloop()

if __name__ == "__main__":
    main()