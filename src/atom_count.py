import pandas as pd
from rdkit import Chem
from collections import Counter

# CSV を読み込み
df = pd.read_csv("datasets/dataset.csv")

atom_counter = Counter()
invalid_smiles = 0

for smi in df["SMILES"].astype(str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_smiles += 1
        continue

    for atom in mol.GetAtoms():
        atom_counter[atom.GetSymbol()] += 1

print("=== Atom Frequency ===")
for atom, cnt in atom_counter.most_common():
    print(f"{atom}: {cnt}")

print(f"\nInvalid SMILES: {invalid_smiles}")
print(f"Total molecules scanned: {len(df)}")