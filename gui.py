import os
import sys
import PySimpleGUI as sg
import tensorflow as tf
from tensorflow.keras.layers import Layer
import argparse
import json
from sqlalchemy import create_engine, asc, desc
from sqlalchemy.orm import sessionmaker
import numpy as np
import pickle
import shutil
from PIL import Image as pi
import base64
from io import BytesIO
import skimage
from datetime import datetime
from scipy import interpolate
import threading

from db_declare import Base, Project, Image, Timeline, Child
from progan import ProgressiveGAN

GAN_TYPE_NONE = -1
GAN_TYPE_WGAN = 0
GAN_TYPE_PROGAN = 1

WINDOW_BIG = {'size': (2500, 1300), 'sample_small': 2, 'sample_big': 1}
WINDOW_SMALL = {'size': (1200, 1000), 'sample_small': 4, 'sample_big': 2}

IM_GALLERY_SIZE_BROWSER = (5, 4)
IM_GALLERY_SIZE_TIMELINE = (5, 8)
IM_CHILDREN_SIZE = (8, 4)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_PATH, 'gui.db')
PLACEHOLDER_IM_PATH = os.path.join(BASE_PATH, 'placeholder.png')

TIMELINE_INTERP_LINEAR = 'linear'
TIMELINE_INTERP_CUBIC = 'cubic'
TIMELINE_INTERP_SLERP = 'slerp'

timeline_export_done = 0.0


class GAN:
    def __init__(self, project_path):
        model_path = os.path.join(project_path, 'generator.h5')
        label_path = os.path.join(project_path, 'labels.txt')

        self.labels = np.loadtxt(label_path, dtype=str)

        self.num_labels = self.labels.shape[0]
        self.z_dim = 100

        self.generator = tf.keras.models.load_model(model_path, compile=False)

        print(self.generator.summary())

    @staticmethod
    def one_hot(labels, num_labels):
        one_hot_labels = np.eye(num_labels, dtype=np.float32)[labels]
        one_hot_labels = np.reshape(one_hot_labels, [-1, 1, 1, num_labels])
        return one_hot_labels

    @tf.function
    def generate_samples(self, model, z, y):
        return model([z, y], training=False)

    def generate_image(self, z, y):
        image = self.generate_samples(self.generator, z, y)
        image = np.array(image)
        return image

    def get_label_strings(self):
        return self.labels


def gen_filename(ext):
    return datetime.now().strftime('%Y%d%m-%H%M%S%f') + ext


def init_img_folder(gan, path, project, session):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    if type(gan) == GAN:
        for i in range(gan.num_labels):
            z = np.random.normal(size=[10, 1, 1, gan.z_dim]).astype('float32')
            labels = [i] * 10
            y = gan.one_hot(labels, gan.num_labels)

            images = gan.generate_image(z, y)

            for j, image in enumerate(images):
                image = image * 127.5 + 127.5
                fn = gen_filename('.png')
                tf.io.write_file(os.path.join(path, fn), tf.image.encode_png(tf.cast(image, tf.uint8)))
                new_z = z[j].reshape(-1, 1, 1, gan.z_dim)
                new_y = y[j].reshape(-1, 1, 1, gan.num_labels)
                im = Image(path=os.path.join(path, fn), z=pickle.dumps(new_z), y=pickle.dumps(new_y), project=project)
                session.add(im)
                session.commit()
    elif type(gan) == ProgressiveGAN:
        z = np.random.normal(size=[40, gan.z_dim]).astype('float32')
        images = gan.generate_samples(z)
        for i, image in enumerate(images):
            fn = gen_filename('.png')
            tf.io.write_file(os.path.join(path, fn), tf.image.encode_png(tf.cast(image, tf.uint8)))
            new_z = z[i].reshape(-1, gan.z_dim)
            im = Image(path=os.path.join(path, fn), z=pickle.dumps(new_z), y=pickle.dumps(np.asarray([])), project=project)
            session.add(im)
            session.commit()


def get_image_page(session, project, page, size, conf, timelines=[]):
    offset = page * (size[0] * size[1])
    rows = [[sg.Text('Images:')]]
    
    tl_im_ids = [tl.image.id for tl in timelines]

    for i in range(size[0]):
        ims = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (i * size[1])).limit(size[1]).all()
        row = []
        for j, im in enumerate(ims):
            if im.id in tl_im_ids:
                row.append(sg.Button(image_filename=im.path, key=('-IMAGE-', (i, j)), enable_events=True, image_size=(128, 64), image_subsample=conf['sample_small'], button_color='#FF0000'))
            else:
                row.append(sg.Button(image_filename=im.path, key=('-IMAGE-', (i, j)), enable_events=True, image_size=(128, 64), image_subsample=conf['sample_small'], button_color='#FFE400'))
        #row = [sg.Button(image_filename=im.path, key=('-IMAGE-', (i, j)), enable_events=True, image_size=(128, 64), image_subsample=conf['sample_small']) for j, im in enumerate(ims)]
        rows.append(row)

    cntrl_row = [sg.Button('Previous Page', key='-PREV_PAGE-', enable_events=True),
                 sg.Button('Next Page', key='-NEXT_PAGE-', enable_events=True),
                 sg.Text(f'Current Page: {page + 1}', key='-CURR_PAGE_TEXT-')]
    rows.append(cntrl_row)
    return rows


def update_image_page(session, project, page, window, size, conf, selected_id, timelines=[]):
    ims_added = 0
    offset = page * (size[0] * size[1])
    
    tl_im_ids = [tl.image.id for tl in timelines]

    for y in range(size[0]):
        for x in range(size[1]):
            window[('-IMAGE-', (y, x))].update(image_filename=PLACEHOLDER_IM_PATH, image_size=(128, 64), image_subsample=conf['sample_small'])

    for i in range(size[0]):
        ims = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (i * size[1])).limit(size[1]).all()
        ims_added += len(ims)
        for j, im in enumerate(ims):
            if im.id == selected_id:
                window[('-IMAGE-', (i, j))].update(image_filename=im.path, image_size=(128, 64), image_subsample=conf['sample_small'], button_color='#005CFF')
            elif im.id in tl_im_ids:
                window[('-IMAGE-', (i, j))].update(image_filename=im.path, image_size=(128, 64), image_subsample=conf['sample_small'], button_color='#FF0000')
            else:
                window[('-IMAGE-', (i, j))].update(image_filename=im.path, image_size=(128, 64), image_subsample=conf['sample_small'], button_color='#FFE400')
                

    if ims_added > 0:
        window['-CURR_PAGE_TEXT-'].update(f'Current Page: {page + 1}')
        return True
    else:
        return False


def update_sel_image_browser(session, project, page, data, window, size, control_data, gan, current_z, config):
    offset = page * (size[0] * size[1])
    im = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (data[0] * size[1])).limit(size[1]).all()[data[1]]
    window['-SEL_IMAGE-'].update(filename=im.path, size=(512, 256), subsample=config['sample_big'])
    
    for i in range(size[0]):
        for j in range(size[1]):
            window[('-IMAGE-', (i, j))].update(button_color='#FFE400')
    
    window[('-IMAGE-', (data[0], data[1]))].update(button_color='#005CFF')

    if type(gan) == GAN:
        y = pickle.loads(im.y)
        for i in range(len(y[0][0][0])):
            window[f'-CONTROL_LABEL_{i}-'].update(value=y[0][0][0][i] * 100)
            control_data[0][0][0][i] = y[0][0][0][i]

    current_z = pickle.loads(im.z)

    return im.id, current_z
    
    
def update_sel_image_child(window, session, im, index, current_z, config):
    children = session.query(Child).order_by(asc(Child.id)).all()

    current_z = pickle.loads(children[index].z)

    window['-SEL_IMAGE-'].update(data=im, size=(512, 256), subsample=config['sample_big'])

    return current_z


def update_sel_image_timeline(session, project, page, data, window, size, config, timelines):
    offset = page * (size[0] * size[1])
    im = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (data[0] * size[1])).limit(size[1]).all()[data[1]]
    window['-SEL_IMAGE-'].update(filename=im.path, size=(512, 256), subsample=config['sample_big'])
    tl_im_ids = [tl.image.id for tl in timelines]

    for i in range(size[0]):
        ims = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (i * size[1])).limit(size[1]).all()
        for j, _im in enumerate(ims):
            if _im.id in tl_im_ids:
                window[('-IMAGE-', (i, j))].update(button_color='#FF0000')
            else:
                window[('-IMAGE-', (i, j))].update(button_color='#FFE400')
    
    window[('-IMAGE-', (data[0], data[1]))].update(button_color='#005CFF')
    # y = pickle.loads(im.y)
    # for i in range(len(y[0][0])):
    #     window[f'-CONTROL_LABEL_{i}-'].update(value=y[0][0][i])

    return im.id


def create_children(session, gan, im_id, data, current_z):
    session.query(Child).delete()
    session.commit()

    rand_amt = data[1]
    img_amt = int(data[2])
    rand_mult = data[3]

    y = np.array(data[0]).astype('float32')
    y = y.reshape((1, 1, 1, gan.num_labels))
    z = current_z

    if z is not None:
        z = z.reshape((1, 1, 1, gan.z_dim))

    for i in range(img_amt):
        if z is not None:
            new_z = z + np.random.normal(size=[1, 1, 1, gan.z_dim]).astype('float32') * (rand_amt * rand_mult)
        else:
            new_z = np.random.normal(size=[1, 1, 1, gan.z_dim]).astype('float32')
        
        new_child = Child(z=pickle.dumps(new_z), y=pickle.dumps(y))
        session.add(new_child)
        session.commit()


def show_children(session, gan, window, conf):
    children = session.query(Child).order_by(asc(Child.id)).all()
    
    base64_strs = []

    for y in range(IM_CHILDREN_SIZE[1]):
        for x in range(IM_CHILDREN_SIZE[0]):
            window[('-IMAGE_CHILDREN-', (x, y))].update(image_filename=PLACEHOLDER_IM_PATH, image_size=(128, 64), image_subsample=conf['sample_small'])

    for i in range(len(children)):
        child = children[i]
        x = i % IM_CHILDREN_SIZE[0]
        y = i // IM_CHILDREN_SIZE[0]

        if type(gan) == GAN:
            image = gan.generate_image(z=pickle.loads(child.z), y=pickle.loads(child.y))
            image = image * 127.5 + 127.5
        elif type(gan) == ProgressiveGAN:
            z = pickle.loads(child.z).reshape(-1, gan.z_dim)
            image = gan.generate_samples(z=z)
        else:
            image = None

        image = image.astype(np.uint8)
        image = pi.fromarray(image[0])
        with BytesIO() as output_bytes:
            im = pi.fromarray(skimage.img_as_ubyte(image))
            im.save(output_bytes, 'PNG')
            bytes_data = output_bytes.getvalue()
        base64_str = base64.b64encode(bytes_data)
        base64_strs.append(base64_str)
        window[('-IMAGE_CHILDREN-', (x, y))].update(image_data=base64_str, image_size=(128, 64), image_subsample=conf['sample_small'])
    return base64_strs


def save_image(session, gan, project, data, path):
    index = data[1] * IM_CHILDREN_SIZE[0] + data[0]
    child = session.query(Child).order_by(asc(Child.id)).all()[index]
    # print(pickle.loads(child.z))
    if type(gan) == GAN:
        image = gan.generate_image(z=pickle.loads(child.z), y=pickle.loads(child.y))
        image = image * 127.5 + 127.5
    elif type(gan) == ProgressiveGAN:
        image = gan.generate_samples(z=pickle.loads(child.z).reshape(-1, gan.z_dim))
    else:
        image = None

    fn = gen_filename('.png')
    tf.io.write_file(os.path.join(path, fn), tf.image.encode_png(tf.cast(image[0], tf.uint8)))
    im = Image(path=os.path.join(path, fn), z=child.z, y=child.y, project=project)
    session.add(im)
    session.commit()


def add_image_to_timeline(session, project, im_id, im_pos):
    im = session.query(Image).filter_by(project=project, id=im_id).first()
    new_timeline = Timeline(order=im_pos, project=project, image=im)
    session.add(new_timeline)
    session.commit()
    return session.query(Timeline).filter_by(project=project).order_by(asc(Timeline.order)).all()


def update_timeline(window, timelines, offset, conf):
    tls = timelines[offset: offset + IM_GALLERY_SIZE_TIMELINE[1]]
    for i in range(IM_GALLERY_SIZE_TIMELINE[1]):
        window[('-TIMELINE_IMAGE-', i)].update(filename=PLACEHOLDER_IM_PATH, size=(128, 64), subsample=conf['sample_small'])
        window[('-TIMELINE_ORDER-', i)].update(value='0')

    for i in range(len(tls)):
        window[('-TIMELINE_IMAGE-', i)].update(filename=tls[i].image.path, size=(128, 64), subsample=conf['sample_small'])
        window[('-TIMELINE_ORDER-', i)].update(value=str(tls[i].order))


def slerp_interp(points, step_count):
    def slerp(val, low, high):
        low = low.reshape(-1)
        high = high.reshape(-1)
        omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)
        if so == 0:
            # L'Hopital's rule/LERP
            return (1.0 - val) * low + val * high
        return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

    steps = int(step_count / (len(points) - 1))
    vectors = list()
    for i in range(len(points) - 1):
        ratios = np.linspace(0., 1., num=steps)
        for ratio in ratios:
            v = slerp(ratio, points[i], points[i+1])
            v = v.reshape((1, 1, 1, -1))
            vectors.append(v)
    return np.asarray(vectors)


def interp(points, step_count, method):
    def lin_interp1d(y):
        x = np.linspace(0., 1., len(y))
        f = interpolate.interp1d(x, y)
        xnew = np.linspace(0., 1., step_count)
        return f(xnew)

    def cubic_spline_interp1d(y):
        x = np.linspace(0., 1., len(y))
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.linspace(0., 1., step_count)
        return interpolate.splev(xnew, tck, der=0)

    if points.shape[0] < 4:
        raise ValueError('Too few points for interpolation: need 4, got {}'.format(points.shape[0]))

    if method == 'linear':
        return np.apply_along_axis(lin_interp1d, 0, points)
    elif method == 'cubic':
        return np.apply_along_axis(cubic_spline_interp1d, 0, points)
    elif method == 'slerp':
        return slerp_interp(points, step_count)
    else:
        raise ValueError('Interpolation method does not exist: {}'.format(method))


def export_timeline(gan, ims, frames, loop, interp_method, path, window):
    global timeline_export_done
    window['-TIMELINE_EXPORT-'].update(disabled=True)
    # ims = [tl.image for tl in timelines]
    if loop:
        ims.append(ims[0])

    z_keys = []
    y_keys = []

    for im in ims:
        z_keys.append(pickle.loads(im.z))
        # print(pickle.loads(im.z))
        y_keys.append(pickle.loads(im.y))
    
    z_keys = np.asarray(z_keys)
    y_keys = np.asarray(y_keys)

    z_seq = interp(z_keys, frames, interp_method)
    if type(gan) == GAN:
        y_seq = interp(y_keys, frames, interp_method)
    else:
        y_seq = np.asarray([])

    path = os.path.join(path, datetime.now().strftime("%Y%d%m_%H%M%S"))
    os.makedirs(path)

   # if not os.path.isdir(path):
   #     os.makedirs(path)
   # else:
   #     shutil.rmtree(path)
   #     os.makedirs(path)

    # TODO: combine the below two loops
    # gen_ims = []
    for i in range(z_seq.shape[0]):
        # print(f'generating frame {i}')
        z = z_seq[i].reshape(-1, 1, 1, gan.z_dim)

        if type(gan) == GAN:
            y = y_seq[i].reshape(-1, 1, 1, gan.num_labels)
            gen_im = gan.generate_image(z, y)
            gen_im = gen_im * 127.5 + 127.5
        elif type(gan) == ProgressiveGAN:
            gen_im = gan.generate_samples(z.reshape(-1, gan.z_dim))
        else:
            gen_im = None

        gen_im = np.uint8(gen_im)
        # gen_ims.append(gen_im)
        file_path = os.path.join(path, f'{i:09d}.png')
        # print(f'saving file: {file_path}')
        timeline_export_done = int((i / (z_seq.shape[0] - 1)) * 100)
        window['-TIMELINE_EXPORT_PROGRESS-'].update_bar(timeline_export_done)
        tf.io.write_file(file_path, tf.image.encode_png(tf.cast(gen_im[0], tf.uint8)))

    window['-TIMELINE_EXPORT-'].update(disabled=False)


def make_window1(session, project, gan, im_page, size):
    img_browser_row = get_image_page(session, project, im_page, IM_GALLERY_SIZE_BROWSER, size)
    img_control = []
    if type(gan) == GAN:
        label_strings = gan.get_label_strings()
        for i in range(gan.num_labels):
            cntrl = [sg.Text(label_strings[i]), sg.Slider(range=(-100, 100), resolution=1,
                                                          orientation='horizontal', expand_x=True,
                                                          key=f'-CONTROL_LABEL_{i}-', default_value=0,
                                                          enable_events=True)]
            img_control.append(cntrl)

        img_control.append([sg.Button('Reset', key='-RESET_LABEL-')])
        img_control.append([sg.HorizontalSeparator(pad=((0, 0), (20, 20)))])


    img_control.append([sg.Text('multiplier:'), sg.Slider(range=(1, 10000), resolution=1,
                                                      orientation='horizontal', expand_x=True,
                                                      key='-CONTROL_RMULT-', default_value=1,
                                                      enable_events=True)])
    img_control.append([sg.Text('random:'), sg.Slider(range=(0, 100), resolution=1,
                                                      orientation='horizontal', expand_x=True,
                                                      key='-CONTROL_RANDOM-', default_value=50,
                                                      enable_events=True)])
    img_control.append([sg.Text('children:'), sg.Slider(range=(0, 32), resolution=1,
                                                        orientation='horizontal', expand_x=True,
                                                        key='-CONTROL_CHILDREN-', default_value=10,
                                                        enable_events=True)])
    img_control.append([sg.Button('Create Children', key='-CREATE_CHILDREN-'), sg.Button('Random Cousins', key='-RANDOM_CHILDREN-')])
    img_sel = [[sg.Text('Selected image:')], [sg.Image(key='-SEL_IMAGE-', size=(512, 256), subsample=size['sample_big'])],
               [sg.Button('Save Child', key='-SAVE_CHILD-')]]

    img_children = [[sg.Text('Children:')]]
    for x in range(IM_CHILDREN_SIZE[0]):
        row = [sg.Button(key=('-IMAGE_CHILDREN-', (x, y)), enable_events=True, image_filename=PLACEHOLDER_IM_PATH,
                         image_size=(128, 64), image_subsample=size['sample_small']) for y in range(IM_CHILDREN_SIZE[1])]
        img_children.append(row)

    layout = [[sg.Column(img_sel), sg.Column(img_control, expand_x=True)],
              [sg.Column(img_browser_row), sg.Column(img_children)]]

    return sg.Window('Image Browser', layout, size=size['size'], finalize=True)


def make_window2(session, project, im_page, size, timelines=[]):
    img_sel = [[sg.Text('Selected image:')],
               [sg.Image(key='-SEL_IMAGE-', size=(512, 256), subsample=size['sample_big'])],
               [sg.Button('Add To Timeline', key='-ADD_TO_TIMELINE-', enable_events=True)]]
    img_browser_row = get_image_page(session, project, im_page, IM_GALLERY_SIZE_TIMELINE, size, timelines)
    timeline = []
    for i in range(IM_GALLERY_SIZE_TIMELINE[1]):
        item = sg.Column([[sg.Image(filename=PLACEHOLDER_IM_PATH, size=(128, 64), key=('-TIMELINE_IMAGE-', i), subsample=size['sample_small'])],
                          [sg.InputText(default_text='0', size=10, key=('-TIMELINE_ORDER-', i), enable_events=True)],
                          [sg.Button('Remove', key=('-REMOVE_FROM_TIMELINE-', i), enable_events=True)]])
        timeline.append(item)

    timeline_navigation = [sg.Button('Previous', key='-PREV_TIMELINE-', enable_events=True),
                         sg.Button('Update Order', key='-UPDATE_ORDER-', enable_events=True),
                         sg.Button('Next', key='-NEXT_TIMELINE-', enable_events=True)]

    timeline_controls = [[sg.Text('Interpolation:'), sg.Combo([TIMELINE_INTERP_LINEAR,
                                                               TIMELINE_INTERP_CUBIC,
                                                               TIMELINE_INTERP_SLERP], default_value=TIMELINE_INTERP_LINEAR, key='-TIMELINE_INTERP-', enable_events=True)],
                         [sg.Text('Frames:'), sg.InputText(default_text='1000', key='-TIME_LINE_FRAMES-', enable_events=True)],
                         [sg.Text('Loop:'), sg.Checkbox('', default=True, key='-TIMELINE_LOOP-', enable_events=True)],
                         [sg.Text('Folder: '), sg.In(size=(50, 1), enable_events=True, key='-TIMELINE_EXPORT_PATH-'), sg.FolderBrowse()],
                         [sg.Button('Export', key='-TIMELINE_EXPORT-', enable_events=True), sg.ProgressBar(key='-TIMELINE_EXPORT_PROGRESS-', max_value=100, orientation='horizontal', size_px=(150, 30))]]

    layout = [[sg.Column(img_sel), sg.Column(timeline_controls)], [timeline], [timeline_navigation], [sg.Column(img_browser_row)]]

    return sg.Window('Timeline', layout, size=size['size'], finalize=True)


def main():
    global timeline_export_done
    parser = argparse.ArgumentParser(description='Latent Space GUI')
    parser.add_argument('-p', '--project_path', type=str, help='the path to the project folder, doesnt have to exist yet')
    parser.add_argument('-x', '--size_x', type=int, help='tell the gui what the width of the images is')
    parser.add_argument('--small', dest='is_small', action='store_true')
    parser.add_argument('--large', dest='is_small', action='store_false')
    parser.set_defaults(is_small=True)
    parser.add_argument('--wgan', dest='gan_type', action='store_const', const=GAN_TYPE_WGAN)
    parser.add_argument('--progan', dest='gan_type', action='store_const', const=GAN_TYPE_PROGAN)
    parser.set_defaults(gan_type=GAN_TYPE_NONE)

    args = parser.parse_args()

    if args.gan_type == GAN_TYPE_NONE:
        print('you forgot to specifiy the gan type!...')
        sys.exit()

    project_path = os.path.abspath(os.path.join(os.getcwd(), args.project_path))
    print(project_path)
    if not os.path.isdir(project_path):
        print('you entered the wrong directory')
        sys.exit()

    subsample_size_small = args.size_x // 128
    subsample_size_big = args.size_x // 512
    if args.is_small:
        size = WINDOW_SMALL
        size['sample_small'] = subsample_size_small
        size['sample_big'] = subsample_size_big
    else:
        size = WINDOW_BIG
        size['sample_small'] = subsample_size_small // 2
        size['sample_big'] = subsample_size_big // 2

    # print(size)

    if args.gan_type == GAN_TYPE_WGAN:
        gan = GAN(project_path)
    elif args.gan_type == GAN_TYPE_PROGAN:
        gan = ProgressiveGAN(project_path)
        for path in os.listdir(project_path):
            if path.startswith('res_'):
                gan.load_checkpoint(os.path.join(project_path, path))
    else:
        gan = None

    gen_img_path = os.path.join(project_path, 'images', 'generated')
    # timeline_img_path = os.path.join(project_path, 'images', 'timeline')

    engine = create_engine(f'sqlite:///{DATABASE_PATH}')
    Base.metadata.bind = engine
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    im_page_browser = 0
    im_page_timeline = 0
    im_sel_id_browser = 0
    im_sel_id_timeline = 0
    timeline_offset = 0
    timeline_interp = TIMELINE_INTERP_LINEAR
    timeline_frames = 1000
    timeline_loop = True
    control_data = [[[[0. for _ in range(gan.num_labels)]]], 0.5, 10, 1]
    im_sel_child = None
    children_ims = []
    export_path = ''
    current_z = []

    project = session.query(Project).filter_by(path=project_path).first()
    if project is None:
        project = Project(path=project_path)
        session.add(project)
        session.commit()

    timelines = session.query(Timeline).filter_by(project=project).order_by(asc(Timeline.order)).all()
    im_sel_id_timeline_pos = len(session.query(Timeline).filter_by(project=project).all())

    images = session.query(Image).filter_by(project=project).all()

    if len(images) == 0:
        print('initializing images folder...')
        init_img_folder(gan, gen_img_path, project, session)

    sg.theme('DarkAmber')

    # # Create the Windows
    window2, window1= make_window2(session, project, im_page_timeline, size, timelines), make_window1(session, project, gan, im_page_browser, size)

    update_timeline(window2, timelines, timeline_offset, size)

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        window, event, values = sg.read_all_windows()

        if window == sg.WIN_CLOSED:
            break
        if event == sg.WIN_CLOSED or event == 'Exit':
            window.close()

        if len(event) == 2:
            if event[0] == '-IMAGE-':
                if window == window1:
                    im_sel_id_browser, current_z = update_sel_image_browser(session, project, im_page_browser, event[1], window, IM_GALLERY_SIZE_BROWSER, control_data, gan, current_z, size)
                elif window == window2:
                    im_sel_id_timeline = update_sel_image_timeline(session, project, im_page_timeline, event[1], window, IM_GALLERY_SIZE_TIMELINE, size, timelines)

            if event[0] == '-IMAGE_CHILDREN-':
                im_sel_child = event[1]
                # print(im_sel_child)
                # print(im_sel_child[1] * IM_CHILDREN_SIZE[0] + im_sel_child[0])
                current_z = update_sel_image_child(window, session, children_ims[im_sel_child[1] * IM_CHILDREN_SIZE[0] + im_sel_child[0]], im_sel_child[1] * IM_CHILDREN_SIZE[0] + im_sel_child[0], current_z, size)

            if event[0] == '-TIMELINE_ORDER-':
                if values[event] != '':
                    timelines[event[1] + timeline_offset].order = int(values[event])
                    session.commit()

            if event[0] == '-REMOVE_FROM_TIMELINE-':
                print('removing: ' + str(event[1] + timeline_offset))
                removed_item = timelines.pop(event[1] + timeline_offset)
                print(removed_item)
                session.delete(removed_item)
                session.commit()
                update_timeline(window, timelines, timeline_offset, size)

        else:
            if event == '-NEXT_PAGE-':
                if window == window1:
                    new_page = im_page_browser + 1
                    if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_BROWSER, size, im_sel_id_browser):
                        im_page_browser = new_page
                elif window == window2:
                    new_page = im_page_timeline + 1
                    if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_TIMELINE, size, im_sel_id_timeline, timelines):
                        im_page_timeline = new_page

            if event == '-PREV_PAGE-':
                if window == window1:
                    new_page = im_page_browser - 1
                    if im_page_browser <= 0:
                        im_page_browser = 0
                    else:
                        if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_BROWSER, size, im_sel_id_browser):
                            im_page_browser = new_page
                elif window == window2:
                    new_page = im_page_timeline - 1
                    if im_page_timeline <= 0:
                        im_page_timeline = 0
                    else:
                        if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_TIMELINE, size, im_sel_id_timeline, timelines):
                            im_page_timeline = new_page

            if event == '-CONTROL_RANDOM-':
                control_data[1] = float(values['-CONTROL_RANDOM-']) / 100.0

            if event == '-CONTROL_RMULT-':
                control_data[3] = float(values['-CONTROL_RMULT-'])
            
            if event == '-CONTROL_CHILDREN-':
                control_data[2] = values['-CONTROL_CHILDREN-']

            if event.startswith('-CONTROL_LABEL_'):
                data = float(values[event]) / 100.0
                index = int(event.replace('-CONTROL_LABEL_', '').replace('-', ''))
                control_data[0][0][0][index] = data

            if event == '-CREATE_CHILDREN-':
                # print(control_data)
                create_children(session, gan, im_sel_id_browser, control_data, current_z)
                children_ims = show_children(session, gan, window, size)

            if event == '-RANDOM_CHILDREN-':
                create_children(session, gan, im_sel_id_browser, control_data, None)
                children_ims = show_children(session, gan, window, size)
            
            if event == '-SAVE_CHILD-':
                save_image(session, gan, project, im_sel_child, gen_img_path)
                update_image_page(session, project, im_page_browser, window, IM_GALLERY_SIZE_BROWSER, size, im_sel_id_browser)

            if event == '-RESET_LABEL-':
                for i in range(gan.num_labels):
                    window[f'-CONTROL_LABEL_{i}-'].update(value=0)
                    control_data[0][0][0][i] = 0

            if event == '-ADD_TO_TIMELINE-':
                timelines = add_image_to_timeline(session, project, im_sel_id_timeline, im_sel_id_timeline_pos)
                im_sel_id_timeline_pos += 1
                update_timeline(window, timelines, timeline_offset, size)

            #if event == '-REMOVE_FROM_TIMELINE-':
            #   print('remove from timeline')

            if event == '-PREV_TIMELINE-':
                if not timeline_offset <= 0:
                    timeline_offset -= 1
                    update_timeline(window, timelines, timeline_offset, size)

            if event == '-NEXT_TIMELINE-':
                timeline_offset += 1
                update_timeline(window, timelines, timeline_offset, size)

            if event == '-UPDATE_ORDER-':
                timelines = session.query(Timeline).filter_by(project=project).order_by(asc(Timeline.order)).all()
                update_timeline(window, timelines, timeline_offset, size)

            if event == '-TIMELINE_INTERP-':
                timeline_interp = values[event]

            if event == '-TIME_LINE_FRAMES-':
                timeline_frames = int(values[event])

            if event == '-TIMELINE_LOOP-':
                timeline_loop = values[event]

            if event == '-TIMELINE_EXPORT_PATH-':
                export_path = values['-TIMELINE_EXPORT_PATH-']
                # print(values['-TIMELINE_EXPORT_PATH-'])

            if event == '-TIMELINE_EXPORT-':
                # print(export_path)
                if export_path != "":
                    print(timeline_interp)
                    worker = threading.Thread(target=export_timeline, args=(gan, [tl.image for tl in timelines], timeline_frames, timeline_loop, timeline_interp, export_path, window))
                    worker.start()

                    # while timeline_export_done < 1.0:
                    #     print(f'saving timeline: {timeline_export_done}')
                    # export_timeline(gan, timelines, timeline_frames, timeline_loop, timeline_interp, export_path)


if __name__ == '__main__':
    main()
