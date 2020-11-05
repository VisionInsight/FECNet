from pyflann import *
import numpy as np
import flask
import web_util
import random
import logging
import cv2
import math
app = flask.Flask(__name__)

test_path = "/data00/chenriwei/PublicData/FaceData/RAF/RAF"

def norm_feature(features):
    result = []
    s = 0.0
    for f in features:
        s += f*f
    norm = math.sqrt(s)
    for f in features:
        result.append(f/norm)
    return result

flann = FLANN()

with open("feature2.txt") as f:
    dataset = []
    dataset_names = []
    testset = []
    testset_names = []
    cnt = 0
    for line in f:
        cnt +=1
        if cnt > 1000000:
            break
        item = line.split("\t")
        filename, feature = item
        feature = eval(feature)
        if "test" in filename:
            testset.append(norm_feature(feature))
            testset_names.append(filename)
        else:
            dataset.append(norm_feature(feature))
            dataset_names.append(filename)
    dataset = np.array(dataset)
    params = flann.build_index(dataset, algorithm="kmeans", branching=32, iterations=7, checks=16)
    testset = np.array(testset)
    result, dists = flann.nn_index(testset, 6, checks=params["checks"]);

@app.route('/')
def main():
        select_idx = random.randint(0, len(result))
        filenames = []
        filenames.append("empty.jpg")
        filenames.append(testset_names[select_idx])
        filenames.append("empty.jpg")
        for idx in result[select_idx]:
            filenames.append(dataset_names[idx])
        image_htmls = []
        for idx,file_ in enumerate(filenames):
            full_path = os.path.join(test_path, file_)
            image = cv2.imread(full_path)
            if image is None:
                continue
            image_html = web_util.embeding_image_2_string(image)
            image_htmls.append(image_html)
        return web_util.make_image_html(image_htmls, 500)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    port = int(sys.argv[1])
    app.run(host='0.0.0.0', port = port)
