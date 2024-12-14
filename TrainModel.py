import wandb
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import yaml
from pathlib import Path


#download dataset

from roboflow import Roboflow
rf = Roboflow(api_key="uI2mywSlhIEy3Gr7pimT")
project = rf.workspace("medeniyet-universty").project("lastbac")
version = project.version(4)
dataset = version.download("yolov8")

data_bac = '' #yaml file
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

# Tune hypherparameters
results = model.tune(data=data_bac, epochs=20, iterations=100, plots=True, save=False, val=False)
wandb.log(results)


# choose best model
best_yaml = ""
best=""
best_param = yaml.safe_load(Path(best_yaml).read_text())

#Fine tuning the best
model = YOLO(best)
results = model.train(data=data_bac, epochs=200, imgsz=640, patience=100,
                      lr0=best_param['lr0'],
                      lrf=best_param['lrf'],
                      momentum=best_param['momentum'],
                      weight_decay=best_param['weight_decay'],
                      warmup_epochs=best_param['warmup_epochs'],
                      warmup_momentum=best_param['warmup_momentum'],
                      box=best_param['box'],
                      cls=best_param['cls'],
                      dfl=best_param['dfl'],
                      hsv_h=best_param['hsv_h'],
                      hsv_s=best_param['hsv_s'],
                      hsv_v=best_param['hsv_v'],
                      translate=best_param['translate'],
                      fliplr=best_param['fliplr'],
                      mosaic=best_param['mosaic'])

#modeli eğitmiş olduk artık bu modeli tahmin etmek için kullanabiliriz.