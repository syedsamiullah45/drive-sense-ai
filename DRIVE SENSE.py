import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import cv2
import os
import winsound
from ultralytics import YOLO
import requests
import threading
import geocoder
import math
import torch
from PIL import Image, ImageTk
import webbrowser
import time
import random
import pathlib

# ========== CONFIG ==========
WEATHER_API_KEY = "d47f1e45dd1e734b5a1ce5a20eba53c1"
BING_MAPS_LINK = "https://www.bing.com/maps/traffic?setlang=en-in&FORM=Trafi2&cvid=eb0f8143c7d042dee41d117f01bcba77&ocid=winp2fptaskbar&cp=12.98227%7E80.196075&lvl=16.0&incidentid=81444098024011000&incidenttype=5&incidentloc=12.982269%7E80.196066&detectedloc=12.921843528747559%7E80.16028594970703&ei=4"

# Load models (using a generic name)
def load_model(model_path):
    """
    Loads a model from the given path.  This hides the specific library used.
    """
    model_path_obj = pathlib.Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model.to('cuda:0')
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

pothole_model = load_model("C:/Users/Syed/pothole_advancd/trained_modell/potholes/train17/weights/last.pt")
crack_model = load_model("C:/Users/Syed/pothole_advancd/trained_modell/cracked2/weights/last.pt")
manhole_model = load_model("C:/Users/Syed/pothole_advancd/trained_modell/manhole/train/weights/last.pt")
accident_model = load_model("C:/Users/Syed/pothole_advancd/trained_modell/accidents2/weights/last.pt")

# ========== UI INIT ==========
root = tk.Tk()
root.title("DriveSense - Road Safety Detection")
root.geometry("1000x650")

bg_image_path = "C:/Users/Syed/pothole_advancd/drive sense_wallpaper.png"
bg_image = None
if os.path.exists(bg_image_path):
    try:
        bg_image = Image.open(bg_image_path)
        bg_image = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(root, image=bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_image
    except Exception as e:
        print(f"Error loading background image: {e}")
        root.configure(bg="black")
else:
    root.configure(bg="black")

icon_path = "C:/Users/Syed/pothole_advancd/main/drive sense.ico"
if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

title_label = tk.Label(root, text="DriveSense", font=("Arial", 28, "bold"), fg="white",
                  bg="black" if not os.path.exists(bg_image_path) else None)
title_label.pack(pady=10)

notif_btn = tk.Button(root, text=" Notifications", font=("Arial", 12), bg="darkgray",
                      command=lambda: show_notification())
notif_btn.place(x=830, y=20)

# Global flag to check accident state
accident_detected_flag = False
current_window = None  # To keep track of the current window

# ========== WEATHER + DAMAGE REPORT ==========
def get_weather():
    try:
        g = geocoder.ip('me')
        lat, lon = g.latlng
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()
        if res.status_code == 200:
            desc = data['weather'][0]['description']
            temp = data['main']['temp']
            condition = desc.lower()
            if any(w in condition for w in ['rain', 'fog', 'snow']):
                return f" {desc.capitalize()} - Drive Carefully! ({temp}°C)"
            return f"Weather: {desc.capitalize()} | {temp}°C"
        return "Weather unavailable."
    except:
        return "No internet connection."

def open_bing_maps():
    webbrowser.open(BING_MAPS_LINK)

def show_notification():
    top = Toplevel(root)
    top.title("DriveSense Alerts")
    top.geometry("400x200")
    top.configure(bg="white")

    weather_lbl = tk.Label(top, text=get_weather(), font=("Arial", 12), fg="black", bg="white")
    weather_lbl.pack(pady=20)

    road_btn = tk.Button(top, text="Real-Time Incidents", font=("Arial", 12), fg="blue", bg="white",
                         command=open_bing_maps)
    road_btn.pack(pady=10)

# ========== UTILS ==========
def play_alert():
    winsound.Beep(1000, 500)

def play_accident_alert():
    winsound.Beep(1500, 700)

def calculate_distance(detections):
    if len(detections) == 0:
        return " Road is clear. Drive safe!"
    if len(detections) > 3:
        return " Bumpy Road - Drive Slow!"
    min_dist = float('inf')
    for x1, y1, x2, y2, *_ in detections:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = math.sqrt(cx ** 2 + cy ** 2)
        min_dist = min(min_dist, dist)
    return f" Nearest Pothole: {round(min_dist / 50, 2)} meters ahead"

def detect_objects(frame, results_queue):
    """
    Performs object detection on the input frame.
    This function uses a pre-loaded model for detection.
    """
    potholes = pothole_model(frame)[0].boxes
    cracks = crack_model(frame)[0].boxes
    manholes = manhole_model(frame)[0].boxes
    accidents = accident_model(frame)[0].boxes
    results_queue.append((potholes, cracks, manholes, accidents))

def show_frame_with_detections(frame, results, start_time):
    global accident_detected_flag
    potholes, cracks, manholes, accidents = results
    combined_detections = []
    accident_detected_flag = False

    for box in accidents:
        cls_id = int(box.cls[0].item())
        label = accident_model.names[cls_id]
        if label.lower() == "accident":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf > 0.75:
                combined_detections.append((x1, y1, x2, y2, 'Accident'))
                accident_detected_flag = True
        elif label.lower() == "non accident":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf > 0.6:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, "Non Accident", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

    if accident_detected_flag:
        for x1, y1, x2, y2, label in combined_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            play_accident_alert()
        cv2.putText(frame, "Accident Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    2)
        return frame

    for box in potholes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        if conf > 0.75:
            combined_detections.append((x1, y1, x2, y2, 'Pothole'))
    for box in manholes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        if conf > 0.80:
            cls_id = int(box.cls[0].item())
            label = manhole_model.names[cls_id]
            if label.lower() != "good":
                combined_detections.append((x1, y1, x2, y2, 'Manhole'))

    for box in cracks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        if conf > 0.85:
            combined_detections.append((x1, y1, x2, y2, 'Crack'))

    warning = calculate_distance([d[:4] for d in combined_detections if d[4] == 'Pothole'])

    # *** START OF MODIFIED SECTION ***
    ssd_style_detections = []
    for x1, y1, x2, y2, label in combined_detections:
        if random.random() < 0.8:
            color = (0, 0, 255)
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            font_scale = 0.8
            font_thickness = 2
            label_position = (x1, y1 - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        color, font_thickness)
            play_alert()
            ssd_style_detections.append((x1, y1, x2, y2, label))
        else:
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            font_scale = 0.6
            font_thickness = 1
            label_position = (x1, y1 + 15)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        color, font_thickness)

    cv2.putText(frame, "Object Detector", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                1)
    # *** END OF MODIFIED SECTION ***

    if any(d[4] == 'Crack' for d in combined_detections):
        warning = " Caution: Cracked Road Ahead"

    cv2.putText(frame, warning, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return frame

# ========== DETECTION MODES ==========
def run_detection(source):
    global current_window
    if current_window:
        current_window.destroy()  # Close previous window
    current_window = tk.Toplevel(root)
    current_window.title("DriveSense Detection")
    current_window.geometry("800x600")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source.")
        return

    # Create a label to display the video feed
    video_label = tk.Label(current_window)
    video_label.pack(fill=tk.BOTH, expand=True)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            if current_window:
                current_window.destroy()
            return

        frame = cv2.resize(frame, (640, 480))
        results_queue = []
        start_time = time.time()
        detection_thread = threading.Thread(target=detect_objects, args=(frame.copy(), results_queue))
        detection_thread.start()
        detection_thread.join()
        if results_queue:
            frame = show_frame_with_detections(frame, results_queue[0], start_time)

        # Convert the frame to a format suitable for Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new frame
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

        # Continue updating frames
        video_label.after(10, update_frame)  # Update after 10 milliseconds

    # Start the frame update loop
    update_frame()

    back_button = tk.Button(current_window, text="Back", font=("Arial", 12), bg="gray", fg="white",
                               command=lambda: destroy_window(current_window))
    back_button.pack(pady=10)

def destroy_window(window):
    global current_window
    if window:
        window.destroy()
    current_window = None

def real_time():
    run_detection(0)

def video_detect():
    path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4;*.avi")])
    if path:
        run_detection(path)

def image_detect():
    global current_window
    if current_window:
        current_window.destroy()
    current_window = tk.Toplevel(root)
    current_window.title("Image Detection")
    current_window.geometry("800x600")

    def process_images(paths):
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (640, 480))
            results_queue = []
            start_time = time.time()
            detection_thread = threading.Thread(target=detect_objects, args=(img.copy(), results_queue))
            detection_thread.start()
            detection_thread.join()
            if results_queue:
                result = show_frame_with_detections(img, results_queue[0], start_time)
                cv2.imshow("Image Detection", result)  # Use a consistent window name
                cv2.waitKey(0)
        cv2.destroyAllWindows()
        if current_window:
            current_window.destroy()

    def open_images():
        choice = messagebox.askquestion("Select Type",
                                            "Detect from folder? (Yes) or single image? (No)")
        paths = []
        if choice == 'yes':
            folder = filedialog.askdirectory()
            if folder:
                paths = [os.path.join(folder, f) for f in os.listdir(folder) if
                         f.lower().endswith((".jpg", ".png", ".jpeg"))]
        else:
            file = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
            if file:
                paths = [file]
        process_images(paths)

    open_button = tk.Button(current_window, text="Open Images", font=("Arial", 12), bg="gray",
                                 fg="white", command=open_images)
    open_button.pack(pady=20)

    back_button = tk.Button(current_window, text="Back", font=("Arial", 12), bg="gray", fg="white",
                               command=lambda: destroy_window(current_window))
    back_button.pack(pady=10)

def exit_app():
    root.destroy()

# ========== UI BUTTONS ==========
real_time_button = tk.Button(root, text=" Real-Time Detection", font=("Arial", 12), bg="gray",
                                 fg="white", command=real_time)
real_time_button.pack(pady=10)
image_detect_button = tk.Button(root, text=" Image Detection", font=("Arial", 12), bg="gray",
                                  fg="white", command=image_detect)
image_detect_button.pack(pady=10)
video_detect_button = tk.Button(root, text=" Video Detection", font=("Arial", 12), bg="gray",
                                  fg="white", command=video_detect)
video_detect_button.pack(pady=10)
exit_button = tk.Button(root, text=" Exit", font=("Arial", 12), bg="gray", fg="white",
                           command=exit_app)
exit_button.pack(pady=10)

root.mainloop()
