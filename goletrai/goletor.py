import cv2
import numpy as np
import onnxruntime as ort
import os 
from .tools import Models

cFile = os.path.abspath(__file__)
basepath = os.path.dirname(cFile)
SRCPATH = os.path.join(basepath, "src")
if not os.path.exists(SRCPATH):
    os.mkdir(SRCPATH)

CACHEPATH = os.path.join(SRCPATH, "models.json")
MODELPATH = os.path.join(SRCPATH, "rai.onnx")

if not os.path.exists(MODELPATH):
    Models.update(CACHEPATH, MODELPATH)

class GoletRai:
    def __init__(self, model_path:str=MODELPATH, conf_thresh: float = 0.5, iou_thresh:float=0.5):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
    
    def get_pad(self, image):
        img_height, img_width = image.shape[:2]
        obj_shape = max(img_height, img_width)
        top_pad = (obj_shape - img_height) // 2
        bottom_pad = obj_shape - img_height - top_pad
        left_pad = (obj_shape - img_width) // 2
        right_pad = obj_shape - img_width - left_pad
        image_pad = cv2.copyMakeBorder(
            image, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=[127, 127, 127]
        )
        return image_pad, [img_height, img_width]

    def pre_process(self, image_pad):
        image_pad = cv2.cvtColor(image_pad, cv2.COLOR_BGR2RGB)
        image_pad = cv2.resize(image_pad, (self.input_width, self.input_height))
        input_tensor = np.array(image_pad) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return input_tensor

    def scale_box(self, boxes, ishape):
        gain = min(self.input_height / ishape[0], self.input_width / ishape[1])
        pad = (
            round((self.input_width - ishape[1] * gain) / 2 - 0.1),
            round((self.input_height - ishape[0] * gain) / 2 - 0.1),
        )
        boxes[..., 0] -= pad[0]
        boxes[..., 1] -= pad[1]
        boxes[..., 2] -= pad[0]
        boxes[..., 3] -= pad[1]
        boxes[..., :4] /= gain
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, ishape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, ishape[0])
        return boxes

    def post_process(self, outputs, ishape):
        out_arr = outputs[0][0]
        mask = out_arr[:, 4] >= self.conf_thresh
        filtered = out_arr[mask]
        results = []
        for dfil in filtered:
            bbox = self.scale_box(dfil, ishape)
            x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
            score = float(bbox[4])
            results.append([x1, y1, x2, y2, score])
        return results

    def ning(self, image):
        image_pad, ishape = self.get_pad(image)
        input_tensor = self.pre_process(image_pad)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        results = self.post_process(outputs, ishape)
        return results
    
    def coba(self, conf_thresh:float=0.5, show:bool=False):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = self.ning(image, conf_thresh)
        if len(results) > 0:
            for result in results:
                print(f"BBOX: {result[:4]} - CONF: {result[-1]}")
                if not show: continue
                cv2.rectangle(image, (result[0], result[1]), (result[2], result[3]), (0, 0, 255), 2, 1)
        else:
            print("Wajah Tidak Ditemukan")

        if show:
            cv2.imshow('image', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
    
    def gambar(self, image, results):
        for result in results:
            cv2.rectangle(image, (result[0], result[1]), (result[2], result[3]), (0, 0, 255), 2, 1)
        return image 
    
    @staticmethod
    def update_model():
        Models.update(CACHEPATH, MODELPATH)