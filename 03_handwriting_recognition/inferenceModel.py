import cv2
import numpy as np
import pandas as pd

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.configs import BaseModelConfigs

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

if __name__ == "__main__":
    configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202301111911/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/03_handwriting_recognition/202301111911/val.csv").values.tolist()

    # Replace "image_path" with the actual path to the image file
    image_path = "Datasets/IAM_Words/words/coba/coba-00/banyak_1.png"
    image = cv2.imread(image_path)

    # Find the corresponding label in the CSV file
    label = None
    for csv_image_path, csv_label in df:
        if csv_image_path == image_path:
            label = csv_label
            break

    prediction_text = model.predict(image)
    if label is not None:
        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
        
        # Resize the image for display
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No label found for image: {image_path}, Prediction: {prediction_text}")
        
        # Resize the image for display
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
