'''
By running face detection on a uploaded video, the trained mask detection model can scan the frames that detect said faces
and determine whether the person is wearing a mask.
'''


import click
import cv2
import torch
from skvideo.io import vreader, FFmpegWriter
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from common.facedetector import FaceDetector
from data_training import MaskDetect


@click.command(help="""
                    videoPath: path to video file to annotate\n
                    outputPath: path to save output to
                    """)
@click.argument('videopath')
@click.argument('outputpath')

@torch.no_grad()
def tagVideo(videopath, outputpath):

    # Load the trained mask detection model
    model = MaskDetect()
    model.load_state_dict(torch.load('models/epoch=8.ckpt')['state_dict'], strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load a Ml model that detects human faces
    faceDetect = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )

    # Define the transformations that are performed on the video frames from the uploaded video
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    writer = FFmpegWriter(str(outputpath) + '.avi')

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    # iterate over the frames of the video and detect human faces in every frame
    for frame in vreader(str(videopath)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetect.detect(frame)

        # clamp coordinates for every face in the frame
        for face in faces:
            xStart, yStart, width, height = face

            xStart, yStart = max(xStart, 0), max(yStart, 0)

            faceImg = frame[yStart:yStart + height, xStart:xStart + width]

            # run mask detection on the human faces and make a prediction
            output = model(transformations(faceImg).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)
            print(predicted)

            # draw rectangle/labels around the faces with the mask prediction
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          labelColor[predicted],
                          thickness=5)

            textSize = cv2.getTextSize(labels[predicted], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0]

            cv2.putText(frame,
                        labels[predicted],
                        (textX, yStart - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, labelColor[predicted], 5)

        writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    tagVideo()

