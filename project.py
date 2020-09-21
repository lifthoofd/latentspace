import os
import sys
from shutil import copy
import json
from PIL import Image
import skimage
import base64
from io import BytesIO
import numpy as np

from imgdb_declare import Project as ProjectDb

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_PATH, 'projects')


class Project:
    def __init__(self):
        # project object container, can create new one or load an existing one, a project is every subfolder of a path
        # a project consists of a json file for all hyperparams and file locations and folders for checkpoints, samples, etc
        self.name = None
        self.data = None
        self.project = None

    def new_project(self, name, Session):
        session = Session
        self.name = name

        # check if project already exists, if yes load otherwise create new
        path = os.path.join(PROJECT_DIR, self.name)
        if not os.path.isdir(path):
            # create project folder
            os.mkdir(path)

            # copy json file template
            json_def_path = os.path.join(PROJECT_DIR, 'template.json')
            json_path = os.path.join(path, f'{self.name}.json')
            copy(json_def_path, json_path)

            # load json data in self.data
            with open(json_path, 'r+') as json_file:
                self.data = json.load(json_file)

                ckpt_path = os.path.join(path, 'checkpoints')
                if not os.path.isdir(ckpt_path):
                    os.mkdir(ckpt_path)
                samples_path = os.path.join(path, 'samples')
                if not os.path.isdir(samples_path):
                    os.mkdir(samples_path)
                images_path = os.path.join(path, 'images')
                if not os.path.isdir(images_path):
                    os.mkdir(images_path)

                self.data['path'] = path
                self.data['gan']['ckpt_path'] = ckpt_path
                self.data['gan']['samples_path'] = samples_path
                self.data['gan']['images_path'] = images_path

                json_file.seek(0)
                json.dump(self.data, json_file)
                json_file.truncate()

            new_project = ProjectDb(name=self.name, path=path)
            session.add(new_project)
            session.commit()
            self.project = new_project

        else:
            print('project already exists, loading this one')
            # self.load_project(self.name)

    def load_project(self, idx, Session):
        session = Session()
        # self.name = name

        project = session.query(ProjectDb).filter_by(id=idx).first()

        self.name = project.name
        self.project = project

        path = project.path
        if os.path.isdir(path):
            json_path = os.path.join(path, f'{self.name}.json')
            with open(json_path, 'r') as json_file:
                self.data = json.load(json_file)
        else:
            print('project does not exist, nothing is loaded')

    def save_project(self):
        pass

    def get_project_id(self):
        return self.project.id

    def get_config(self):
        if self.data is not None:
            return self.data

    def get_gan_config(self):
        if self.data is not None:
            return self.data['gan']

    def set_gan_config(self, config):
        json_path = os.path.join(self.data['path'], f'{self.name}.json')

        for key in config:
            self.data['gan'][key] = config[key]

        with open(json_path, 'r+') as json_file:
            json_file.seek(0)
            json.dump(self.data, json_file)
            json_file.truncate()

        print(self.data)

    def get_latest_sample(self):
        # self.data['gan']['samples_path']
        files = os.listdir(self.data['gan']['samples_path'])
        files.sort()

        if len(files) > 0:
            return self.encode_img_path(os.path.join(self.data['gan']['samples_path'], files[-1]))
        else:
            return ''

    # @staticmethod
    def encode_img_path(self, path):
        image = Image.open(path)
        return self.img_to_base64(image)

    def encode_img_array(self, arr):
        # print(arr.shape)
        arr = arr * 127.5 + 127.5
        arr = arr.astype(np.uint8)
        # print(arr)
        image = Image.fromarray(arr)
        # print('enoding')
        return self.img_to_base64(image)

    def img_to_base64(self, image):
        with BytesIO() as output_bytes:
            im = Image.fromarray(skimage.img_as_ubyte(image))
            im.save(output_bytes, 'JPEG')  # Note JPG is not a vaild type here
            bytes_data = output_bytes.getvalue()

        # encode bytes to base64 string
        base64_str = str(base64.b64encode(bytes_data), 'utf-8')
        # print(base64_str)
        return base64_str

