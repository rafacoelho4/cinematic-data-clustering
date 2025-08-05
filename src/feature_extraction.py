import ffmpeg 
import librosa 
import pandas as pd 
import numpy as np 

def audio_features(video_file, video_name, df_shots):
    # Explicar isso aqui 
    (
        ffmpeg
        .input(video_file)
        .output(f'{video_name}.wav', acodec='pcm_s16le', ac=1, ar=22050)
        .overwrite_output()
        .run(quiet=True)
    )

    def analyze_shot_audio(shot_id, start, duration, wav_path='inthemood.wav',
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
        dict(**analyze_shot_audio(rid, row.start, row.duration, wav_path=f'{video_name}.wav'))
        for rid, row in df_shots.iterrows()
    ])
    df_shots = df_shots.merge(audio_feats, on='shot_id')
    
    return df_shots 