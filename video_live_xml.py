from pathlib import Path

import click
import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from common.facedetector import FaceDetector
from data_training import MaskDetect

@click.command(help="""
                    modelPath: path to model.ckpt\n
                    videoPath: path to video file to annotate
                    """)
@click.argument('modelpath')
@torch.no_grad()
def tagVideo(modelpath):
    model = MaskDetect()
    model.load_state_dict(torch.load(modelpath)['state_dict'], strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    faceDetector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    video_capture = cv2.VideoCapture(0)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = faceDetector.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

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

        cv2.imshow('FaceDetection', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tagVideo()
