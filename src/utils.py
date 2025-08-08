import ffmpeg 

def get_video_info(video_file) -> dict:
    """
    Retorna metadados básicos do vídeo:
      - duration (segundos),
      - fps (frames por segundo),
      - largura / altura (pixels),
      - tamanho do arquivo (bytes).
    Usa a API 'ffmpeg.probe', que roda puramente em Python.
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