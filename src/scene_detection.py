import pandas as pd 
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scene(video_name):
    # Inicializa PySceneDetect
    video_manager = VideoManager([f'{video_name}_lowres.mp4'])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))  # limiar padrão

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    # Monta DataFrame com início, fim e duração de cada shot
    df_shots = pd.DataFrame([{
        'shot_id': i,
        'start': scene[0].get_seconds(),
        'end': scene[1].get_seconds()
    } for i, scene in enumerate(scene_list)])
    df_shots['duration'] = df_shots['end'] - df_shots['start']
    print(f"Total de shots: {len(df_shots)}")