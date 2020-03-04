import cv2
import torch

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg


def setup_cfg(config_file, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    return cfg
    

class Detector(object):
    def __init__(self, config_file="E:/DL/det2/YJH/faster_rcnn_R_50_FPN_1x.yaml", thres=0.5, class_names=["nc", "pgw"]):
        self.class_names = class_names
        self.cpu_device = torch.device("cpu")
        cfg = setup_cfg(config_file, thres)
        self.predictor = DefaultPredictor(cfg)
        print("Init Model: ", config_file, cfg.MODEL.WEIGHTS, thres, class_names)
        
        self.image = None
        self.font_size = 0.5
        self.font_bold = 2
        
    def detect(self, img, keep_best=False):
        """
        Args:
            keep_best: keep the most confidence object or all objects
        Returns:
            objects (list): the output of the model.
        """
        if type(img) == str:
            self.image = cv2.imread(img)
            if self.image is None:  
                print("image_path cannot open...")
                return None
        else:
            self.image = img
        try:
            predictions = self.predictor(self.image)
        except:
            print("predict failure!")
            return None
        
        instances = predictions["instances"].to(torch.device("cpu"))
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes if instances.has("pred_classes") else None
        labels = [self.class_names[i] for i in classes]
        objects=[]
        if len(labels)==0:  return objects
        
        #print(int(torch.argmax(scores)))    #置信度最高的object index
        indexes = [int(torch.argmax(scores))] if keep_best else range(0, len(scores))
        #for label, score, box in zip(labels, scores, boxes):
        for i in indexes:
            label, score, box = labels[i], scores[i], boxes[i].tensor[0]
            #print(box)
            box = list(map(int, box.tolist()))
            score = score.tolist()
            objects.append([label, score, int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            self._drawPred(label, score, int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        return objects
        
        
    def _drawPred(self, label, conf, left, top, right, bottom):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        font_size = 0.5 if w / 3600 < 0.5 else w / 3600
        font_bold = 2 if font_size > 1 else 1
        # Draw a bounding box.
        cv2.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255), 4)
        text = '{0:} {1:.2f}'.format(label, conf)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_bold)
        top = max(top, labelSize[1])
        cv2.rectangle(self.image, (left, top - round(1.3*labelSize[1])), (left + round(labelSize[0]), top + baseLine), (0, 0, 0), cv2.FILLED)
        cv2.putText(self.image, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, font_size, (180,180,80), font_bold)
    
    
    def save(self, img_path):
        if self.image is not None:
            cv2.imwrite(img_path, self.image)
    
    
    def show(self, winName="Result"):
        if self.image is not None:
            cv2.imshow(winName, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
        
if __name__ == "__main__":   
    jpg = "E:/DL/det2/1111.jpg"
    
    detector = Detector(class_names=['yw', 'yw'])
    objects = detector.detect(jpg, keep_best=False)
    print(objects)
    detector.show()
    detector.save("result.jpg")