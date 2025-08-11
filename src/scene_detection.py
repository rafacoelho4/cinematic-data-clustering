import pandas as pd 
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os 
import ffmpeg

def detect_scene(video_file):
    # Inicializa PySceneDetect
    file_path = f'data/{video_file}'
    if(os.path.exists(file_path)):
        video_manager = VideoManager([f'data/{video_file}'])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))  # limiar padrão

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    else: 
        print("Video file does not exist.", video_file)
        return pd.DataFrame()

    # Monta DataFrame com início, fim e duração de cada shot
    df = pd.DataFrame([{
        'shot_id': i,
        'start': scene[0].get_seconds(),
        'end': scene[1].get_seconds()
    } for i, scene in enumerate(scene_list)])
    df['duration'] = df['end'] - df['start']
    print(f"Total de shots: {len(df)}")
    
    return df 

# Gera um video com menor resolução para acelerar detecção de cenas 
def detect_scene_low_resolution(video_file):
    """
    Detecta as cenas de um filme/video. Utiliza o vídeo em menor resolução para acelerar processo. 
    Se vídeo em menor resolução não existe, ele é gravado primeiro e depois são detectadas as cenas. 

    Arugumentos:
        video_file (string): caminho até arquivo .mp4, do diretório raiz. 

    Retorna:
        DataFrame: colunas 'shot_id', 'start', 'end' e 'duration'.
    """

    video_name = video_file[:-4] 
    # Verifica se ja existe video de menor resolução. Se sim, usar para detectar as cenas
    file_path = f'data/{video_name}_lowres.mp4' 
    if(os.path.exists(file_path)): 
        return detect_scene(f'{video_name}_lowres.mp4') 
    else: 
        # Se não, gera um vídeo a 15 fps e 360 px de altura (largura ajustada automaticamente) 
        try:
            (ffmpeg
            .input(f'data/{video_file}')
            .filter('fps', fps=15)
            .filter('scale', -2, 360)
            .output(f'data/{video_name}_lowres.mp4')
            .overwrite_output()
            .run(quiet=True))
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode('utf8'))

        return detect_scene(f'{video_name}_lowres.mp4')