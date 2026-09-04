
import os

dossier = "../TestPapier/dataset" 


fichiers = [f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))]

for f in fichiers:
    print(f'"{f}",')
