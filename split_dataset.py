import os
import shutil
import random
from pathlib import Path
 
# ---------- CONFIGURACIÓN ----------
RAW_DIR = Path("HOJAS")              # tu carpeta real con las fotos por clase
SPLIT_DIR = Path("dataset_split")    # carpeta de salida (se crea sola)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10                    # el resto, no hace falta usarla directamente
SEED = 42                            # para que la división sea reproducible
 
# Extensiones de imagen válidas (por si el celular guarda en otro formato)
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
 
# ------------------------------------
 
def get_images(class_dir: Path):
    return [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXT]
 
 
def split_list(files, train_ratio, val_ratio, seed):
    files = files[:]  # copia para no mutar la original
    random.Random(seed).shuffle(files)
 
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
 
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
 
    return train_files, val_files, test_files
 
 
def copy_files(files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)
 
 
def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"No encuentro la carpeta '{RAW_DIR}'. "
            f"Revisa que estés corriendo el script desde 'LeafScan_MDD' "
            f"(donde también está la carpeta HOJAS)."
        )
 
    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
 
    if not class_dirs:
        raise ValueError(f"No encontré subcarpetas de clase dentro de {RAW_DIR}")
 
    print(f"Clases encontradas: {[d.name for d in class_dirs]}\n")
 
    resumen = []
 
    for class_dir in class_dirs:
        clase = class_dir.name
        images = get_images(class_dir)
        total = len(images)
 
        if total == 0:
            print(f"⚠️  {clase}: 0 imágenes, se omite.")
            continue
 
        train, val, test = split_list(images, TRAIN_RATIO, VAL_RATIO, SEED)
 
        copy_files(train, SPLIT_DIR / "train" / clase)
        copy_files(val, SPLIT_DIR / "val" / clase)
        copy_files(test, SPLIT_DIR / "test" / clase)
 
        resumen.append((clase, total, len(train), len(val), len(test)))
 
    # ---------- Reporte final ----------
    print(f"{'Clase':30} {'Total':>6} {'Train':>7} {'Val':>6} {'Test':>6}")
    print("-" * 60)
    for clase, total, n_train, n_val, n_test in resumen:
        print(f"{clase:30} {total:>6} {n_train:>7} {n_val:>6} {n_test:>6}")
 
    total_general = sum(r[1] for r in resumen)
    print("-" * 60)
    print(f"Total de imágenes procesadas: {total_general}")
    print(f"\nListo. Revisa la carpeta: {SPLIT_DIR.resolve()}")
 
 
if __name__ == "__main__":
    main()