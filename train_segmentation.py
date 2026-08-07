import os
import sys
import requests
from pathlib import Path
from ultralytics import YOLO

BACKEND_RELOAD_URL = "http://localhost:8000/reload-models"

def main():
    dataset_yaml = Path(r"d:\MoyaProyectoHojas\HojitasdeCacao\data\cacao\data.yaml").resolve()
    
    # 1. Configurar rutas de entrenamiento en data.yaml
    cacao_dir = dataset_yaml.parent

    yaml_content = f"""path: {cacao_dir.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['Healthy Leaf', 'Iron', 'Potassium']
"""
    with open(dataset_yaml, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"data.yaml verificado: {dataset_yaml}")

    # 2. Cargar modelo preentrenado de Segmentación de YOLO26
    print("Cargando modelo YOLO26 Segment preentrenado...")
    model = YOLO("yolo26n-seg.pt")

    # 3. Entrenar y guardar pesos nativos PyTorch (.pt)
    print("Iniciando entrenamiento de Segmentación (Guardando pesos nativos .pt)...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=15,
        imgsz=224,
        batch=16,
        name="cacao_seg_run",
        project=r"d:\MoyaProyectoHojas\HojitasdeCacao\runs",
        exist_ok=True,
    )

    pt_best_path = Path(r"d:\MoyaProyectoHojas\HojitasdeCacao\runs\cacao_seg_run\weights\best.pt")
    print("\n=======================================================")
    print(f"ENTRENAMIENTO FINALIZADO CON ÉXITO")
    print(f"Pesos nativos PyTorch (.pt) guardados en: {pt_best_path}")
    print("=======================================================\n")

    # 4. Notificar a la API de backend (backend/main.py) para recargar los nuevos pesos inmediatamente
    try:
        print(f"Notificando al Servidor Backend ({BACKEND_RELOAD_URL}) para recargar modelos en memoria...")
        res = requests.post(BACKEND_RELOAD_URL, timeout=5)
        if res.status_code == 200:
            print("✅ Backend actualizado con los nuevos pesos del entrenamiento.")
        else:
            print(f"El backend respondió con código {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Servidor Backend no activo o inaccesible (Inicia backend/main.py para usar los nuevos pesos): {e}")

if __name__ == "__main__":
    main()
