import ffmpeg 
import librosa 
import pandas as pd 
import numpy as np 
import os 
import cv2 

# Frame extraction 
def extract_frame_mid(shot_id, time, video_name, input_path='data/snippet.wav'):
    
    output = f'data/frames/{video_name[:5]}_{shot_id:03d}_{time:.3f}.jpg'
    # Skip extraction if file already exists
    if os.path.exists(output):
        return output
    
    try:
        (ffmpeg
        .input(input_path, ss=time)
        .output(output, vframes=1, format='mjpeg', qscale='2')
        .overwrite_output()
        .run(quiet=True))
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode('utf8'))

    return output

def extract_frames(video_file, df):
    """
    Extrai um frame por cena presente no DataFrame. O frame escolhido é aquele exatamente no meio do tempo da cena. 

    Arugumentos:
        video_file (string): caminho até arquivo .mp4, do diretório raiz.        
        df (DataFrame): com coluna obrigatória 'shot_id', 'start' e 'end'. 

    Retorna:
        DataFrame: incrementado com coluna 'frame_path'.
    """
    video_name = video_file[:-4]
    os.makedirs('data/frames', exist_ok=True)
    
    df['frame_path'] = df.apply(lambda row: extract_frame_mid(
        int(row.shot_id),
        (row.start + row.end) / 2,
        video_name,
        f"data/{video_file}"
    ), axis=1)

    return df

def compute_hsv_hist_stats(frame_path): 
    img = cv2.imread(frame_path) 
    if img is None:
        print(f"Erro ao carregar imagem: {frame_path}")
        return {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Conversão para RGB (para os novos histogramas)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)
    
    # Cálculo dos histogramas HSV (existente)
    hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])  
    
    # Cálculo dos histogramas RGB (novos)
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    
    # Normalização
    hist_h = hist_h.flatten() / hist_h.sum() 
    hist_s = hist_s.flatten() / hist_s.sum() 
    hist_v = hist_v.flatten() / hist_v.sum() 
    hist_r = hist_r.flatten() / hist_r.sum() 
    hist_g = hist_g.flatten() / hist_g.sum() 
    hist_b = hist_b.flatten() / hist_b.sum() 

    return {
        # Métricas HSV 
        'hue_mean': float(np.sum(np.arange(180) * hist_h)), # hue 
        'hue_var': float(np.sum(((np.arange(180) - np.sum(np.arange(180) * hist_h))**2) * hist_h)),
        'sat_mean': float(np.sum(np.arange(256) * hist_s)), # saturation 
        'sat_var': float(np.sum(((np.arange(256) - np.sum(np.arange(256) * hist_s))**2) * hist_s)),
        'val_mean': float(np.sum(np.arange(256) * hist_v)),  # value (brightness) 
        'val_var': float(np.sum(((np.arange(256) - np.sum(np.arange(256) * hist_v))**2) * hist_v)), 
        
        # Métricas RGB 
        'r_mean': float(np.sum(np.arange(256) * hist_r)),  # Média do canal Vermelho
        'r_var': float(np.sum(((np.arange(256) - np.sum(np.arange(256) * hist_r))**2) * hist_r)),  # Variância do Vermelho
        'g_mean': float(np.sum(np.arange(256) * hist_g)),  # Média do canal Verde
        'g_var': float(np.sum(((np.arange(256) - np.sum(np.arange(256) * hist_g))**2) * hist_g)),  # Variância do Verde
        'b_mean': float(np.sum(np.arange(256) * hist_b)),  # Média do canal Azul
        'b_var': float(np.sum(((np.arange(256) - np.sum(np.arange(256) * hist_b))**2) * hist_b)),  # Variância do Azul
    }

# Image features 
def image_features(df):
    """
    Analisa uma série de atributos dos frames extraidos de cada cena. 

    Arugumentos:
        df (DataFrame): com coluna obrigatória 'frame_path'. 

    Retorna:
        DataFrame: incrementado com atributos de análise de imagem (hsv, rgb...).
    """
    color_feats = pd.DataFrame([
        dict(shot_id=row.shot_id, **compute_hsv_hist_stats(row.frame_path))
        for _, row in df.iterrows()
    ])
    df = color_feats.merge(df, on='shot_id')
    return df
        
# Audio features 
def save_wav(video_file):
    video_name = video_file[:-4]
    
    file_path = f'data/{video_name}.wav'
    if(os.path.exists(file_path)): 
        return 
    else:
        try:
            (ffmpeg
            .input(f"data/{video_file}")
            .output(f'data/{video_name}.wav', acodec='pcm_s16le', ac=1, ar=22050)
            .overwrite_output()
            .run(quiet=True))
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode('utf8'))

def audio_features(video_file, df):

    video_name = video_file[:-4]

    # Explicar isso aqui 
    save_wav(video_file)

    def analyze_shot_audio(shot_id, start, duration, wav_path='data/snippet.wav',
                        frame_length=2048, hop_length=512, n_fft=2048):

        y, sr = librosa.load(wav_path, sr=None, offset=start, duration=duration, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))

        # Definimos silêncio como valores abaixo de 10 % da mediana ou ≤ 0.001
        threshold = max(0.001, np.median(rms) * 0.10)
        silence_ratio = float(np.mean(rms < threshold))

        return {
            'shot_id': shot_id,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'silence_ratio': silence_ratio
        }

    audio_feats = pd.DataFrame([
        dict(**analyze_shot_audio(rid, row.start, row.duration, wav_path=f'data/{video_name}.wav'))
        for rid, row in df.iterrows()
    ])
    df = df.merge(audio_feats, on='shot_id')
    
    return df 

def movement_features(video_file, df):
    def flow_magnitude_between_frames(path, start, end, max_tries=3, debug=False):
        """
        Calcula a magnitude média do movimento visual entre dois frames dentro do intervalo [start, end].
        Retorna NaN se não for possível ler frames válidos.

        Args:
            path (str): caminho do vídeo (ex: 'inthemood.mp4')
            start, end (float): início e fim da cena (em segundos)
            max_tries (int): número máximo de tentativas para ler dois frames
            debug (bool): se True, imprime erros para diagnóstico

        Returns:
            float: magnitude média (movimento visual) ou np.nan
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return np.nan

        duration = end - start
        offsets = [0.1 * duration, 0.5 * duration]  # 10% e 50% dentro da cena
        frames = []
        for offs in offsets:
            t_ms = (start + offs) * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(frames) == 2:
                break

        cap.release()
        if len(frames) < 2:
            return np.nan

        flow = cv2.calcOpticalFlowFarneback(frames[0], frames[1],
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))

    flow_feats = []
    for _, row in df.iterrows():
        mag = flow_magnitude_between_frames(f"data/{video_file}", row.start, row.end)
        flow_feats.append({'shot_id': row.shot_id, 'flow_mag_mean': mag})

    df = df.merge(pd.DataFrame(flow_feats), on='shot_id')

    return df 
