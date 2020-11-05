import sys,os
import json
import random
from flask import Flask
import flask
import numpy as np
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3
    from io import BytesIO
import cv2
import base64


def embeding_image_2_string(image):
    ret,buff = cv2.imencode('.png',image)
    buff = BytesIO(np.array(buff).tostring())
    data = base64.b64encode(buff.getvalue()).decode("utf-8")
    data = data.replace('\n', '')
    image_html = ('data:image/png;base64,' + data)
    return image_html


def make_image_html(image_htmls, size = 300):
    html_str = []
    html_str.append('<html> <head> <style> \
    .c_img{position:relative; height='+str(size)+'px; width='+str(size)+'px; border:1px solid red;} \
    .c_words{position:relative; top:-530px; height:30px; line-height: 30px; color:#FF0000; right: -260px;} \
    </style> </head> <body>\n')
    for image_html in image_htmls:
        html_str.append('<img class="c_img" width="'+str(size)+'" height="'+str(size)+'" \
            src="' +image_html + '">\n')
    html_str.append('</body></html>\n')
    html = "".join(html_str)
    return html

