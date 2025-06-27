import cv2
from ultralytics import YOLO
from src.tracker import update_tracks
from src.utils import blur_region

def process_video(input_path, output_path, model_name='yolov8n.pt', class_ids=[0],
                  treshold=0.25, tracker_type='byte', ksize=51):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    #while True:
    #    ret, frame = cap.read()
    #    if not ret:
    #        break
#
    #    results = model(frame)[0]
    #    tracks = update_tracks(results, frame.shape)
#
    #    for track in tracks:
    #        cls = int(track.class_id)
    #        if cls in class_ids:
    #            x1, y1, x2, y2 = map(int, track.to_ltrb())
    #            frame = blur_region(frame, x1, y1, x2, y2)
#
    #    out.write(frame)

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        results = model(frame)[0]
        detections = update_tracks(results)

        # Parcours les tracks pour flouter
        for (x1, y1, x2, y2), cls_id in zip(detections.xyxy, detections.class_id):
            if int(cls_id) in class_ids:
                frame = blur_region(frame, int(x1), int(y1), int(x2), int(y2))

        out.write(frame)

    cap.release()
    out.release()
    print(f"[✓] Vidéo traitée sauvegardée : {output_path}")
