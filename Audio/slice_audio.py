import os
import re

# Répertoire où se trouvent vos fichiers .wav
# Remplacez par le chemin adéquat si besoin
DIR = "."

# Regex pour matcher "case7.wav", "case8.wav", ..., "case70.wav"
pattern = re.compile(r"^case(\d+)\.wav$", re.IGNORECASE)

# Parcours du dossier
for fname in os.listdir(DIR):
    m = pattern.match(fname)
    if not m:
        continue

    n = int(m.group(1))
    index = n - 7  # case7 → index 0, ..., case70 → index 63

    # On vérifie qu'on est bien entre 0 et 63
    if not (0 <= index < 64):
        print(f"⚠ Ignoré : {fname} (index hors plage)")
        continue

    # calcul de la colonne (0=A,1=B,…,7=H) et de la rangée (0=1ère rangée,…,7=8ème)
    col = index % 8
    row = index // 8

    letter = chr(ord('A') + col)
    number = row + 1

    new_name = f"{letter}{number}.wav"
    old_path = os.path.join(DIR, fname)
    new_path = os.path.join(DIR, new_name)

    # Si un fichier existe déjà, on avertit
    if os.path.exists(new_path):
        print(f"⚠ Le fichier {new_name} existe déjà !")
    else:
        os.rename(old_path, new_path)
        print(f"{fname} → {new_name}")
