import os
import sys
from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
import numpy as np
import pickle
from scipy import signal, interpolate
import PIL.Image
import shutil

from project import Project
from imgdb_declare import Base, Image, Timeline
from imgdb_declare import Project as ProjectDb
# from creataset import creataset
from dcgan.dcgan import DCGAN

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECTS_PATH = os.path.join(BASE_PATH, 'projects')
DATASETS_PATH = os.path.join(BASE_PATH, 'datasets')
LOG_PATH = os.path.join(BASE_PATH, 'logs')

app = Flask("__main__", static_url_path='')
project = Project()
gan = None

db_path = os.path.join(BASE_PATH, 'database/images.db')
engine = create_engine('sqlite:///{}'.format(db_path))
Base.metadata.bind = engine
session_factory = sessionmaker(bind=engine)
Session = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# sys.stdout = open(os.path.join(LOG_PATH, 'ls.log'), "w")


@app.route("/list_projects")
def get_project_list():
    session = Session()
    projects = session.query(ProjectDb).all()
    projects_dicts = []
    for proj in projects:
        projects_dicts.append(proj.get_dict())
    return jsonify({'projects': projects_dicts})


@app.route("/load_project/<int:idx>")
def load_project(idx):
    # print(idx)
    global gan
    # print(pr)
    project.load_project(idx, Session)
    print(project.get_config())
    if gan is None:
        gan = DCGAN(project.get_gan_config())
    else:
        if not gan.is_training:
            gan = DCGAN(project.get_gan_config())
    return jsonify({'project': project.get_config()})
    # return jsonify({'project': ''})


@app.route("/new_project/<pr>")
def new_project(pr):
    global gan
    print(pr)
    project.new_project(str(pr), Session)
    print(project.get_config())
    if gan is None:
        gan = DCGAN(project.get_gan_config())
    else:
        if not gan.is_training:
            gan = DCGAN(project.get_gan_config())
    return jsonify({'project': project.get_config()})


@app.route("/list_datasets")
def list_datasets():
    datasets = [{'path': os.path.join(DATASETS_PATH, ds), 'name': ds} for ds in os.listdir(DATASETS_PATH) if ds.endswith('.npz')]
    return jsonify({'datasets': datasets})


@app.route("/start_train", methods=["POST"])
def start_train():
    global gan
    print('start train')
    if request.method == "POST":
        # print(request.json['gan'])
        if not gan.is_training:
            project.set_gan_config(request.json['gan'])
            gan = DCGAN(project.get_gan_config())
            gan.train()

            return jsonify({'isTraining': 0})
        else:
            print('training session in progress')
            return jsonify({'isTraining': 1})


@app.route("/train_stats")
def get_train_stats():
    global gan
    if gan is not None:
        losses = gan.get_losses()
        img = project.get_latest_sample()
        progress = gan.get_progress()
        # print(progress)
        return jsonify({'losses': losses, 'img': img, 'progress': progress, 'isTraining': int(gan.is_training)})

    return jsonify('')


@app.route("/stop_train")
def stop_train():
    global gan
    if gan is not None:
        if not gan.is_training:
            gan.stop_train()
    return jsonify('')


@app.route("/list_images")
def get_images():
    global gan
    images_dicts = []
    if gan is not None:
        session = Session()
        pid = project.get_project_id()
        gan_config = project.get_gan_config()
        proj = session.query(ProjectDb).filter_by(id=pid).first()
        images = session.query(Image).filter_by(project=proj).all()
        if len(images) == 0:
            for j in range(gan.num_labels):
                y = gan.one_hot([j], gan.num_labels)
                for i in range(10):
                    z = np.random.normal(size=[1, 1, 1, gan_config['z_dim']]).astype('float32')
                    new_image = Image(z=pickle.dumps(z), y=pickle.dumps(y), project=proj)
                    session.add(new_image)

            session.commit()
            images = session.query(Image).all()

        for image in images:
            image_dict = image.get_dict()
            img = gan.generate_image(image_dict['z'], image_dict['y'])
            image_dict['base64'] = project.encode_img_array(img[0])
            image_dict['z'] = image_dict['z'].tolist()
            image_dict['y'] = image_dict['y'].tolist()
            images_dicts.append(image_dict)

    return jsonify({'images': images_dicts})


@app.route("/update_image", methods=['POST'])
def update_image():
    global gan

    if gan is not None:
        if request.method == 'POST':
            session = Session()
            idx = int(request.json['id'])
            z = np.asarray(request.json['z']).astype('float32')
            y = np.asarray(request.json['y']).astype('float32')
            rand = float(request.json['random'])

            image = session.query(Image).filter_by(id=idx).first()
            # print(image)
            image.z = pickle.dumps(z)
            image.y = pickle.dumps(y)
            session.commit()

            img = gan.generate_image(z, y)
            base64 = project.encode_img_array(img[0])
            img_dict = image.get_dict()
            img_dict['z'] = img_dict['z'].tolist()
            img_dict['y'] = img_dict['y'].tolist()
            img_dict['random'] = rand
            img_dict['base64'] = base64

            return jsonify({'singleImage': img_dict})


@app.route("/generate_children", methods=['POST'])
def generate_children():
    global gan
    gan_config = project.get_gan_config()
    image_dicts = []
    if gan is not None:
        if request.method == "POST":
            z = np.array(request.json['z']).astype('float32')
            y = np.array(request.json['y']).astype('float32')
            rand = float(request.json['random'])
            amnt = int(request.json['amount'])
            for i in range(amnt):
                # do something with the rand and z values here
                z_mod = np.random.normal(size=[1, 1, 1, gan_config['z_dim']]).astype('float32') * rand
                new_z = z + z_mod
                img = gan.generate_image(new_z, y)
                base64 = project.encode_img_array(img[0])
                image_dict = {'z': new_z.tolist(), 'y': y.tolist(), 'base64': base64}
                image_dicts.append(image_dict)

    return jsonify({'children': image_dicts})


@app.route("/add_image", methods=['POST'])
def add_image():
    if request.method == 'POST':
        session = Session()
        z = np.array(request.json['z'])
        y = np.array(request.json['y'])
        proj = session.query(ProjectDb).filter_by(id=project.get_project_id()).first()
        new_image = Image(z=pickle.dumps(z), y=pickle.dumps(y), project=proj)
        session.add(new_image)
        session.commit()

        return jsonify({'image': ''})


@app.route("/add_to_timeline", methods=['POST'])
def add_to_timeline():
    if request.method == 'POST':
        session = Session()
        if 'id' in request.json['image']:
            image = session.query(Image).filter_by(id=request.json['image']['id']).first()
        else:
            z = np.array(request.json['image']['z'])
            y = np.array(request.json['image']['y'])
            proj = session.query(ProjectDb).filter_by(id=project.get_project_id()).first()
            image = Image(z=pickle.dumps(z), y=pickle.dumps(y), project=proj)
            session.add(image)
            session.commit()

        new_timeline = Timeline(image=image, project=image.project)
        session.add(new_timeline)
        session.commit()

    return jsonify({'image': ''})


@app.route("/list_timeline")
def list_timeline():
    global gan
    session = Session()
    proj = session.query(ProjectDb).filter_by(id=project.get_project_id()).first()
    tl_images = session.query(Timeline).filter_by(project=proj).order_by(Timeline.order).all()
    image_dicts = []
    for image in tl_images:
        image_dict = image.image.get_dict()
        img = gan.generate_image(image_dict['z'], image_dict['y'])
        base64 = project.encode_img_array(img[0])
        image_dict['order'] = image.order
        ret = {'imgId': image_dict['id'], 'base64': base64, 'order': image.order}
        image_dicts.append(ret)

    return jsonify({'images': image_dicts})


@app.route("/update_order", methods=['POST'])
def update_order():
    if request.method == 'POST':
        session = Session()
        tl_image = session.query(Timeline).filter_by(image_id=request.json['id']).first()
        print(request.json['order'])
        tl_image.order = request.json['order']
        session.commit()

        return jsonify({'order': ''})


def cubic_spline_interp(points, step_count):
    def cubic_spline_interp1d(y):
        x = np.linspace(0., 1., len(y))
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.linspace(0., 1., step_count)
        return interpolate.splev(xnew, tck, der=0)
    if points.shape[0] < 4:
        raise ValueError('Too few points for cubic interpolation: need 4, got {}'.format(points.shape[0]))
    return np.apply_along_axis(cubic_spline_interp1d, 0, points)


@app.route("/export_frames", methods=['POST'])
def export_frames():
    global gan
    if request.method == 'POST':
        frames = int(request.json['frames'])
        loop = request.json['loop']
        session = Session()
        proj = session.query(ProjectDb).filter_by(id=project.get_project_id()).first()
        tl_images = session.query(Timeline).filter_by(project=proj).order_by(Timeline.order).all()
        gan_config = project.get_gan_config()
        if loop:
            tl_images.append(tl_images[0])
        z_keys = []
        y_keys = []

        for image in tl_images:
            image_dict = image.image.get_dict()
            z_keys.append(image_dict['z'][0])
            y_keys.append(image_dict['y'][0])

        z_keys = np.asarray(z_keys)
        y_keys = np.asarray(y_keys)

        z_seq = cubic_spline_interp(z_keys, frames)
        y_seq = cubic_spline_interp(y_keys, frames)

        z_seqs = z_seq.reshape([-1, 100, 1, 1, gan_config['z_dim']])
        y_seqs = y_seq.reshape([-1, 100, 1, 1, gan.num_labels])

        images = []
        for i in range(z_seqs.shape[0]):
            ims = gan.generate_image(z_seqs[i], y_seqs[i])
            images.append(ims)

        images = np.asarray(images)
        images = images.reshape([-1, gan_config['image_size'], gan_config['image_size'], 3])

        path = os.path.join(gan_config['images_path'], 'timeline')
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

        for i, img in enumerate(images):
            file_path = os.path.join(path, str(i + 1).zfill(8) + '.png')
            print(f'saving: {file_path}')

            img = img * 127.5 + 127.5
            img = np.uint8(img)
            im = PIL.Image.fromarray(img)
            im.save(file_path, format='PNG', quality=100)

        return jsonify({'export': ''})


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if request.method == 'POST':
        for key in request.files:
            file = request.files[key]
            path = os.path.join(DATASETS_PATH, file.name)
            print(f'saved to {path}')
            file.save(path)

        return jsonify({'file_uploaded': ''})


app.run(host='0.0.0.0', debug=True)
