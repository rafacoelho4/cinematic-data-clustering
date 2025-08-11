import ffmpeg 
from pathlib import Path
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

def project_root():
    return Path(__file__).resolve().parent.parent

def from_root(relative_path):
    return project_root() / relative_path

def get_video_info(video_file) -> dict:
    """
    Retorna metadados básicos do vídeo:
      - duration (segundos),
      - fps (frames por segundo),
      - largura / altura (pixels),
      - tamanho do arquivo (bytes).

    Arugumentos:
        video_file (string): caminho até arquivo .mp4, do diretório raiz.         
    """
    try:
        info = ffmpeg.probe(f'data/{video_file}') 
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode('utf8'))

    # Stream de vídeo geralmente é o primeiro elemento em 'streams' com codec_type='video'
    vs = [s for s in info['streams'] if s.get('codec_type')=='video'][0]
    duration = float(vs.get('duration') or info['format'].get('duration') or 0)
    width  = int(vs.get('width', 0))
    height = int(vs.get('height', 0))

    # Frame rate vem como algo tipo '24000/1001' → parse float
    num, den = map(float, vs.get('r_frame_rate','0/1').split('/'))
    fps = num/den if den else 0
    size_bytes = int(info['format'].get('size', 0))
    
    return {
        'duration': duration,
        'fps': fps,
        'width': width,
        'height': height,
        'size_bytes': size_bytes
    }

def cut_scenes(df, min, max):
    """
    Exclui, do DataFrame de cenas, aquelas com duração menor que min e maior que max. 

    Arugumentos:
        df (DataFrame): com coluna obrigatória 'duration'. 
        min (int): tempo minimo, em segundos, de duração das cenas a serem mantidas. 
        max (int): tempo máximo, em segundos, de duração das cenas a serem mantidas. 

    Retorna:
        DataFrame: cenas que estão dentro do intervalo de duração desejada.
    """

    cut_df = df[df['duration'] > min] 
    cut_df = cut_df[cut_df['duration'] < max] 

    return cut_df 