import os
import torchaudio
import pandas as pd
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
import numpy as np

AUDIO_FOLDER = r"C:\Users\Rishi S Etagi\Desktop\medivoice\LIBRI"

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

wav_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".flac") or f.endswith(".wav")]

data = []

for fname in tqdm(wav_files, desc="Extracting embeddings"):
    fpath = os.path.join(AUDIO_FOLDER, fname)
    try:
        signal, fs = torchaudio.load(fpath)
        emb = classifier.encode_batch(signal).squeeze().numpy()
        row = [fname] + emb.tolist()
        data.append(row)
    except Exception as e:
        print(f"Failed to process {fname}: {e}")

columns = ['filename'] + [f'embedding_{i}' for i in range(192)]
df = pd.DataFrame(data, columns=columns)

df.to_csv("ecapa_embeddings.csv", index=False)
print("Saved embeddings to ecapa_embeddings.csv")
