import json

from flask import Flask, render_template, request, redirect, flash
import psycopg2
import base64
import numpy as np
import cv2
import requests

app = Flask(__name__, static_folder="frontend/static", template_folder="frontend/templates")
conn = psycopg2.connect(
    host="ml.n19",
    database="tdbm",
    user="web1",
    password="123456"
)


def memoryview_to_bytes_for_image(mw):
    return base64.b64encode(base64.b64decode(bytes(mw))).decode()


def image_to_jpeg_nparray(image, quality=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    is_success, im_buf_arr = cv2.imencode(".jpg", image, quality)
    return im_buf_arr


def resize_image(image, desired_width):
    current_width = image.shape[1]
    scale_percent = desired_width / current_width
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def image_to_jpeg_bytes(image, quality=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    buf = image_to_jpeg_nparray(image, quality)
    byte_im = buf.tobytes()
    return byte_im


def compress_image(image, grayscale=True, desired_width=480):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = resize_image(image, desired_width)
    return image


def optimize_to_send(image, target_size=480):
    reduced = image.copy()
    if target_size is not None:
        reduced = compress_image(reduced, desired_width=target_size)
    byte_im = image_to_jpeg_bytes(reduced)
    img_enc = base64.b64encode(byte_im).decode("utf-8")
    img_dump = {"img_base64": img_enc}
    return img_dump


def result_from_response(response):
    dct = json.loads(response.content.decode('utf-8'))
    assert dct.get('success')
    assert dct.get('predictions') is not None
    return dct['predictions']['embedding']


def process_rec_face_request(image, target_size=480):
    send_ = optimize_to_send(image, target_size)
    response = requests.post('http://ml.n19:11080/embed_face_2', data=send_)
    embedding = result_from_response(response)
    return embedding


def get_name_by_id_and_tablename(id, tb_name):
    cur = conn.cursor()
    cur.execute(f"SELECT name FROM public.{tb_name} WHERE uid = '{id}'")
    name = cur.fetchone()[0]
    cur.close()
    return name


def changed_view_time(time_str):
    return time_str[:-13]


def update_db(db):
    result = []
    for ind, row in enumerate(db):
        image = memoryview_to_bytes_for_image(row[4])
        event_time = row[5].strftime("%d.%m.%Y %H:%M:%S")
        if row[1] is None:  # Если пользователь не идентифицирован
            plate = "" if row[3] is None else row[3]
            result.append(("", image, event_time, plate))
        elif row[3] is None:  # Если пользователь не на машине
            name = get_name_by_id_and_tablename(row[1], "faces")
            result.append((name, image, event_time, ""))
        else:  # Если на машине
            name = get_name_by_id_and_tablename(row[1], "plates")
            result.append((name, image, event_time, row[3]))
    return result


@app.route('/')
def index():
    cur = conn.cursor()
    cur.execute('SELECT * FROM public.events ORDER BY timestamp DESC LIMIT 10;')
    db = cur.fetchall()
    db = update_db(db)
    cur.close()
    return render_template("index.html", db=db)


@app.route('/addPlate', methods=['post'])
def add_plate():
    plate = request.form.get('plate')
    fio = request.form.get('fio')
    cur = conn.cursor()
    cur.execute(f"INSERT INTO public.plates (name, plate) VALUES ('{fio}', '{plate}');")
    conn.commit()
    cur.close()
    return redirect("/")


@app.route('/addFace', methods=['post'])
def add_face():
    fio = request.form.get('fio')
    nparr = np.frombuffer(request.files.get('face').read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    embedding = process_rec_face_request(img_np)
    if embedding:
        cur = conn.cursor()
        cur.execute(f"INSERT INTO public.faces (name, embedding) VALUES ('{fio}', '{embedding}');")
        conn.commit()
        cur.close()
    return redirect("/")


if __name__ == '__main__':
    app.run()
    conn.close()
