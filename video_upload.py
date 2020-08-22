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
    """ detect if persons in video are wearing masks or not
    """
    model = MaskDetect()
    model.load_state_dict(torch.load('models/epoch=8.ckpt')['state_dict'], strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    faceDetector = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    writer = FFmpegWriter(str(outputpath) + '.avi')

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    for frame in vreader(str(videopath)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetector.detect(frame)
        for face in faces:
            xStart, yStart, width, height = face

            xStart, yStart = max(xStart, 0), max(yStart, 0)

            faceImg = frame[yStart:yStart + height, xStart:xStart + width]
            output = model(transformations(faceImg).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)
            print(predicted)

            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          labelColor[predicted],
                          thickness=5)

            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0]

            cv2.putText(frame,
                        labels[predicted],
                        (textX, yStart - 20),
                        font, 2, labelColor[predicted], 5)

        writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    tagVideo()

