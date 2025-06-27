import datetime
from src.blur import process_video

# Tu peux changer ici selon le type d'objet à flouter :
# COCO classes : 0 = personne, 2 = voiture, 5 = bus, 7 = camion, etc.
if __name__ == "__main__":
    filename = 'extrait_villejourrapide'  # Nom du fichier vidéo sans extension
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    input_path = f'data/pixels/{filename}.mp4'
    output_path = f'results/output_{timestamp}_{filename}.mp4'

    #confiance yolo
    #treshold = 0.15  # seuil de confiance pour les détections
    #param tracker

    #model dispo
    #yolov8n.pt très petit
    #yolo11n.pt petit
    #yolo11.pt moyen
    #yolo11l.pt grand
    #yolo11x.pt très grand

    process_video(input_path, output_path, model_name="yolo11x.pt", class_ids=[0,2],
                   tracker_type='byte', ksize=51)
