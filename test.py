import os
import numpy as np
import torch
from models.FECNet import FECNet
from models.mtcnn import MTCNN
from PIL import Image
import cv2

def test():
    model = FECNet()
    model.load_state_dict(torch.load('data/FECNet.pt'))
    model.eval()
    with torch.no_grad():
        test_path = "/data00/chenriwei/PublicData/FaceData/RAF/RAF"
        with open("feature2.txt",'w') as fout:
            idx = 1
            for filename in os.listdir(test_path):
                full_path = os.path.join(test_path, filename)
                try:
                    img = Image.open(full_path)
                    channels = img.split()
                    if len(channels) != 3:
                        continue
                    r,g,b = channels
                    img = Image.merge("RGB", (b, g, r))
                    mtcnn = MTCNN(image_size=224)
                except:
                    # ignore all exception
                    continue
                face, prob = mtcnn(img, return_prob=True)
                idx +=1
                face = np.array(face)
                if face.any():
                    face = torch.Tensor(face).view(1,3,224,224)
                    Embedding = model(face.cuda()).cpu()
                    fout.write("{}\t{}\n".format(filename, list(Embedding[0,:].numpy())))



if __name__ == "__main__":
    test()
