import pandas as pd 
import numpy as np 

import moviepy.editor as mpy 
import subprocess 

import os 
import warnings 
import subprocess 

def write_video(df,
    video_file,
    cid,
    output_dir="data/highlight",
    fps=10,
    resize_factor=0.5,
    bitrate="500k"):
    
    shots = df.sort_values('start').copy()

    video_path = os.path.abspath(video_file)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado em {video_path}")

    # Filtra timestamps inválidos
    shots = shots.dropna(subset=['start','end'])
    shots['start'] = shots['start'].astype(float)
    shots['end']   = shots['end'].astype(float)
    
    if shots.empty:
        warnings.warn(f"DataFrame não tem cenas válidas; pulando.")
        return

    print("shot:", shots['start'])
    temp_files = []
    for idx, row in shots.iterrows():
        s, e = row.start, row.end
        if e <= s:
            warnings.warn(f"Cena {idx} inválida (start ≥ end: {s} ≥ {e}); pulando.")
            continue

        tmp_fp = os.path.join(output_dir, f"tmp_c{cid}_{idx}.mp4")
        # Extrai subclip — s e e são floats garantidos
        clip = mpy.VideoFileClip(video_path).subclip(s, e).resize(resize_factor)
        clip.write_videofile(
            tmp_fp,
            fps=fps,
            codec="libx264",
            preset="ultrafast",
            bitrate=bitrate,
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        clip.close()
        temp_files.append(tmp_fp)

    if not temp_files:
        warnings.warn(f"Nenhum subclip gerado para cluster {cid}.")
        return 

    # 3) Concatena sem re-encodar
    list_fp = os.path.join(output_dir, f"list_c{cid}.txt")
    with open(list_fp, "w") as f:
        for tmp in temp_files:
            f.write(f"file '{tmp}'\n")

    out_fp = os.path.join(output_dir, f"{video_file[:4]}_cluster_{cid}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_fp, "-c", "copy", out_fp
    ], check=True)

    # 4) Limpeza
    for tmp in temp_files:
        os.remove(tmp)
    os.remove(list_fp)

    print(f"Vídeo salvo em: {out_fp}")


def write_top2_cluster_videos(
    df,
    video_file, 
    output_dir="data/highlight",
    fps=10,
    resize_factor=0.5,
    bitrate="500k"):
    """
    Para os 2 clusters com mais cenas em df, gera vídeos contendo todas as cenas de cada cluster.
    video_file é usado tal como passado (relativo a CWD ou absoluto).
    """
    # 1) Resolve caminhos relativos via CWD
    video_path = os.path.abspath(video_file)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado em {video_path}")

    print("video path:", video_path)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 2) Top-2 clusters
    top2 = df['cluster'].value_counts().nlargest(2).index.tolist()

    for cid in top2:
        write_video(df, video_file, cid)
        
        # shots = df[df['cluster']==cid].sort_values('start').copy()
        # # Filtra timestamps inválidos
        # shots = shots.dropna(subset=['start','end'])
        # shots['start'] = shots['start'].astype(float)
        # shots['end']   = shots['end'].astype(float)
        
        # if shots.empty:
        #     warnings.warn(f"Cluster {cid} não tem cenas válidas; pulando.")
        #     continue

        # print("shot:", shots['start'])
        # temp_files = []
        # for idx, row in shots.iterrows():
        #     s, e = row.start, row.end
        #     if e <= s:
        #         warnings.warn(f"Cena {idx} inválida (start ≥ end: {s} ≥ {e}); pulando.")
        #         continue

        #     tmp_fp = os.path.join(output_dir, f"tmp_c{cid}_{idx}.mp4")
        #     # Extrai subclip — s e e são floats garantidos
        #     clip = mpy.VideoFileClip(video_path).subclip(s, e).resize(resize_factor)
        #     clip.write_videofile(
        #         tmp_fp,
        #         fps=fps,
        #         codec="libx264",
        #         preset="ultrafast",
        #         bitrate=bitrate,
        #         audio_codec="aac",
        #         verbose=False,
        #         logger=None
        #     )
        #     clip.close()
        #     temp_files.append(tmp_fp)

        # if not temp_files:
        #     warnings.warn(f"Nenhum subclip gerado para cluster {cid}.")
        #     continue

        # # 3) Concatena sem re-encodar
        # list_fp = os.path.join(output_dir, f"list_c{cid}.txt")
        # with open(list_fp, "w") as f:
        #     for tmp in temp_files:
        #         f.write(f"file '{tmp}'\n")

        # out_fp = os.path.join(output_dir, f"{video_file[:4]}_cluster_{cid}.mp4")
        # subprocess.run([
        #     "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        #     "-i", list_fp, "-c", "copy", out_fp
        # ], check=True)

        # # 4) Limpeza
        # for tmp in temp_files:
        #     os.remove(tmp)
        # os.remove(list_fp)

        # print(f"Vídeo cluster {cid} salvo em: {out_fp}")
