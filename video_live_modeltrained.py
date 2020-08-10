import cv2
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from data_training import MaskDetect
from common.facedetector import FaceDetector

from flask import Flask, Response

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)

@torch.no_grad()
def tagVideo():

    model = MaskDetect()
    model.load_state_dict(torch.load('models/epoch=8.ckpt')['state_dict'], strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    faceDetect = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = faceDetect.detect(gray)

        for face in faces:
            xStart, yStart, width, height = face
            xStart, yStart = max(xStart, 0), max(yStart, 0)

            faceImg = frame[yStart:yStart + height, xStart:xStart + width]
            output = model(transformations(faceImg).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)

            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)

            textSize = cv2.getTextSize(labels[predicted], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2

            cv2.putText(frame,
                        labels[predicted],
                        (textX, yStart - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, labelColor[predicted], 2)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(tagVideo(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
