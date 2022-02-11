import os
os.environ['LRU_CACHE_CAPACITY'] = '1'
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import detectron2

# import some common libraries
import numpy as np
from tqdm import tqdm
import os, json, cv2, random
from PIL import Image
from pathlib import Path
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode

from detectron2.data import MetadataCatalog, DatasetCatalog
from IPython.display import display
from detectron2.engine import DefaultTrainer

from config import layoutComponentDetectionModelDir, layoutComponentDetectionModelDataDir, layoutComponentDetectionCatg2Index, layoutComponentDetectionModelDir


class LayoutComponentDetector(object):

    def __init__(self,status) -> None:
        # data dir
        self.status = status
        self.catg2index =  layoutComponentDetectionCatg2Index
        self.data_dir = layoutComponentDetectionModelDataDir
        print("Creating metadata for model config................")
        if status == 'train':
            DatasetCatalog.register('train', get_data_dicts)
            MetadataCatalog.get('train').set(thing_classes=['Headings'])
            self.train_metadata = MetadataCatalog.get("train")
        elif self.status == 'val':
            DatasetCatalog.register('val', get_data_dicts)
            MetadataCatalog.get('val').set(thing_classes=['Headings'])
            self.valid_meatadata = MetadataCatalog.get("val")

        print('Creating model config....................')
        # define model config
        self.cfg = get_cfg()
        # set model loding to CPU mode
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATALOADER.NUM_WORKERS = 6
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025  
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.OUTPUT_DIR = str(layoutComponentDetectionModelDir)
        # # load the trained model
        # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        # print('Loading model................')
        # self.predictor = DefaultPredictor(self.cfg)

    def traing(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        # uncomment below to train
        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()


    def predict_on_single_image(self, img_pth:str, debug=False):
        """
        Method to run object deteciton model on a single image
        """
        
        im = cv2.imread(img_pth)
        outputs = self.predictor(im.copy()) 
        if debug:
            v = Visualizer(im[:, :, ::-1],
                        metadata=self.valid_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            Image.fromarray(out.get_image()[:, :, ::-1].astype(np.uint8)).save('objectDetectionResult.jpg')

        pred_boxes = outputs.get("instances").get_fields()["pred_boxes"].tensor.cpu().numpy()
        scores = outputs.get("instances").get_fields()["scores"].cpu().numpy()
        pred_classes = outputs.get("instances").get_fields()["pred_classes"].cpu().numpy()

        del outputs
        del im

        return {
            "pred_boxes": pred_boxes,
            "scores": scores,
            "pred_classes": pred_classes
        }


def get_data_dicts():
    dataset_dicts = []
    for idx, img_pth in tqdm(
        enumerate((layoutComponentDetectionModelDataDir / 'train').glob("*.png"))
    ):
        json_file = img_pth.parent / f"{img_pth.stem}.json"
        with open(json_file, "r") as json_handle:
            json_data = json.load(json_handle)
        record = {}

        filename = str(img_pth)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in json_data["shapes"]:
            x, y = list(zip(*anno["points"]))
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            poly = []
            for i in range(len(x)):
                poly.append(x[i])
                poly.append(y[i])

            obj = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == "__main__":
    detctr = LayoutComponentDetector(status='train')
    detctr.traing()
    
