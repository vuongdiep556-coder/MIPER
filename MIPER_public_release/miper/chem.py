from __future__ import annotations

import signal
from contextlib import contextmanager
import pickle
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

Chem = None
AllChem = None
MolFromSmiles = None


def _ensure_rdkit() -> None:
    """Import RDKit only when molecular features must be built.

    Long fine-tuning runs only need cached molecular arrays.  Keeping RDKit out
    of the hot training process reduces C-extension surface area and avoids
    server-specific instability observed when RDKit, pandas and CUDA were all
    resident for many hours.
    """

    global Chem, AllChem, MolFromSmiles
    if Chem is not None:
        return
    from rdkit import Chem as _Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem as _AllChem
    from rdkit.Chem import MolFromSmiles as _MolFromSmiles

    RDLogger.DisableLog("rdApp.*")
    Chem = _Chem
    AllChem = _AllChem
    MolFromSmiles = _MolFromSmiles

# Molecular feature caches may be generated under NumPy 2.x, whose pickles
# reference ``numpy._core``.  The project Python 3.10 environment uses NumPy
# 1.24, where the equivalent module is ``numpy.core``.
if "numpy._core" not in sys.modules and hasattr(np, "core"):
    sys.modules["numpy._core"] = np.core


@contextmanager
def time_limit(seconds: int):
    def _raise_timeout(signum, frame):
        raise TimeoutError("RDKit 3D conformer generation timed out")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def one_hot(value: int, choices: list[int]) -> list[bool]:
    if value not in choices:
        value = choices[-1]
    return [value == choice for choice in choices]


def atom_features(atom: Chem.Atom, one_hot_formal_charge: bool = True) -> np.ndarray:
    attrs: list[bool | int] = []
    attrs += one_hot(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    attrs += one_hot(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    attrs += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if one_hot_formal_charge:
        attrs += one_hot(atom.GetFormalCharge(), [-1, 0, 1])
    else:
        attrs.append(atom.GetFormalCharge())
    attrs.append(atom.IsInRing())
    attrs.append(atom.GetIsAromatic())
    return np.asarray(attrs, dtype=np.float32)


def featurize_smiles(
    smiles: str,
    add_dummy_node: bool = True,
    conformer_mode: str = "2d",
    conformer_timeout: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_rdkit()
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    if conformer_mode == "2d":
        AllChem.Compute2DCoords(mol)
    elif conformer_mode == "3d":
        try:
            with time_limit(conformer_timeout):
                mol_h = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_h, maxAttempts=100)
                AllChem.UFFOptimizeMolecule(mol_h)
            mol = Chem.RemoveHs(mol_h)
        except (Exception, TimeoutError):
            AllChem.Compute2DCoords(mol)
    else:
        raise ValueError(f"Unknown conformer_mode: {conformer_mode}")

    node_features = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    adjacency = np.eye(mol.GetNumAtoms(), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    conf = mol.GetConformer()
    coords = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((diff * diff).sum(axis=-1)).astype(np.float32)

    if add_dummy_node:
        feat = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1), dtype=np.float32)
        feat[0, 0] = 1.0
        feat[1:, 1:] = node_features
        node_features = feat

        adj = np.zeros((adjacency.shape[0] + 1, adjacency.shape[1] + 1), dtype=np.float32)
        adj[1:, 1:] = adjacency
        adjacency = adj

        dist = np.full((distance.shape[0] + 1, distance.shape[1] + 1), 1e6, dtype=np.float32)
        dist[1:, 1:] = distance
        distance = dist

    return node_features, adjacency, distance


def load_or_build_features(
    smiles: list[str],
    cache_path: str | Path | None = None,
    conformer_mode: str = "2d",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cache = pickle.load(f)
            except Exception as exc:
                warnings.warn(
                    f"Ignoring incompatible molecular feature cache {cache_path}: {exc!r}. "
                    "It will be rebuilt for the current Python/NumPy/RDKit environment.",
                    RuntimeWarning,
                )
                cache = {}
    for smi in sorted(set(smiles)):
        if smi not in cache:
            cache[smi] = featurize_smiles(smi, conformer_mode=conformer_mode)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(cache, f)
    return cache


def pad_array(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    shape = (int(shape[0]), int(shape[1]))
    padded = np.zeros(shape, dtype=np.float32)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded


def collate_molecules(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    max_atoms = max(item["node_features"].shape[0] for item in batch)  # type: ignore[index, union-attr]
    feat_dim = batch[0]["node_features"].shape[1]  # type: ignore[index, union-attr]
    node_features, adjacency, distance = [], [], []
    for item in batch:
        node_features.append(pad_array(item["node_features"], (max_atoms, feat_dim)))  # type: ignore[arg-type]
        adjacency.append(pad_array(item["adjacency"], (max_atoms, max_atoms)))  # type: ignore[arg-type]
        distance.append(pad_array(item["distance"], (max_atoms, max_atoms)))  # type: ignore[arg-type]

    out = {
        "adjacency": torch.tensor(np.asarray(adjacency), dtype=torch.float32),
        "node_features": torch.tensor(np.asarray(node_features), dtype=torch.float32),
        "distance": torch.tensor(np.asarray(distance), dtype=torch.float32),
        "x_control": torch.tensor(np.asarray([item["x_control"] for item in batch]), dtype=torch.float32),
        "x_treated": torch.tensor(np.asarray([item["x_treated"] for item in batch]), dtype=torch.float32),
        "dose": torch.tensor(np.asarray([item["dose"] for item in batch])[:, None], dtype=torch.float32),
        "time": torch.tensor(np.asarray([item["time"] for item in batch])[:, None], dtype=torch.float32),
    }
    for key in ["reactome_target", "hallmark_target"]:
        if key in batch[0]:
            out[key] = torch.tensor(np.asarray([item[key] for item in batch]), dtype=torch.float32)
    return out
