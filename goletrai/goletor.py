import cv2
import numpy as np
import onnxruntime as ort
import os 

cFile = os.path.abspath(__file__)
basepath = os.path.dirname(cFile)
SRCPATH = os.path.join(basepath, "src")
if not os.path.exists(SRCPATH):
    os.mkdir(SRCPATH)

MODELPATH = os.path.join(SRCPATH, "rai.onnx")

if not os.path.exists(MODELPATH):
    import requests
    try:
        MODELURL = 'https://github.com/mhbaji/goletrai/releases/download/v0.1.0-pre/rai.onnx'
        print('Downloading model ...')
        response = requests.get(MODELURL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        block_size = 1024

        with open(MODELPATH, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                percent = downloaded * 100 / total_size
                print(f"\rDownloading: {percent:.2f}%", end="")
        print("\nDownloaded.")
    except Exception as e:
        print(f"error: {e}")

class GoletRai:
    def __init__(self, model_path:str=MODELPATH, iou_thresh:float=0.5):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.iou_thresh = iou_thresh

    def preprocess(self, image):
        img = cv2.resize(image, (self.input_width, self.input_height))
        img = np.array(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def ning(self, image, conf_thresh:float=0.5):
        image_input = self.preprocess(image)
        outputs = self.session.run(None, {self.model_inputs[0].name: image_input})
        bboxes = []
        scores = []
        for outt in outputs[0][0]:
            x1, y1, x2, y2, confs, clsss = outt
            if confs >= conf_thresh:
                bbx = [int(_bbx) for _bbx in [x1, y1, x2, y2]]
                bboxes.append(bbx)
                scores.append(float(confs))

        indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_thresh, self.iou_thresh)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                xmin, ymin, xmax, ymax = bboxes[i]
                results.append([xmin, ymin, xmax, ymax, scores[i]])
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