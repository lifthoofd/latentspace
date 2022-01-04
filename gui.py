import os
import sys
import PySimpleGUI as sg
import tensorflow as tf
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

from db_declare import Base, Project, Image, Timeline, Child
import dcgan

IM_GALLERY_SIZE_BROWSER = (5, 4)
IM_GALLERY_SIZE_TIMELINE = (5, 8)
IM_CHILDREN_SIZE = (5, 4)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_PATH, 'gui.db')
PLACEHOLDER_IM_PATH = os.path.join(BASE_PATH, 'placeholder.png')

TIMELINE_INTERP_LINEAR = 'linear'
TIMELINE_INTERP_SINE = 'sine'


def gen_filename(ext):
    return datetime.now().strftime('%Y%d%m-%H%M%S%f') + ext


def init_img_folder(gan, path, project, session):
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

    for i in range(gan.num_labels):
        z = np.random.normal(size=[10, 1, 1, gan.z_dim]).astype('float32')
        #z = tf.random.normal([10, 1, 1, gan.z_dim])
        labels = [i] * 10
        y = gan.one_hot(labels, gan.num_labels)
        
        print(z.shape)
        print(y.shape)

        images = gan.generate_image(z, y)

        for j, image in enumerate(images):
            image = image * 127.5 + 127.5
            fn = gen_filename('.png')
            tf.io.write_file(os.path.join(path, fn), tf.image.encode_png(tf.cast(image, tf.uint8)))
            new_z = z[j].reshape(-1, 1, 1, gan.z_dim)
            new_y = y[j].reshape(-1, 1, 1, gan.num_labels)
            im = Image(path=os.path.join(path, fn), z=pickle.dumps(new_z), y=pickle.dumps(new_y), project=project)
            #im = Image(path=os.path.join(path, fn), z=pickle.dumps(z[j]), y=pickle.dumps(y[j]), project=project)
            session.add(im)
            session.commit()


def get_image_page(session, project, page, size):
    offset = page * (size[0] * size[1])
    rows = [[sg.Text('Images:')]]

    for i in range(size[0]):
        ims = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (i * size[1])).limit(size[1]).all()
        row = [sg.Button(image_filename=im.path, key=('-IMAGE-', (i, j)), enable_events=True, image_size=(128, 64), image_subsample=2) for j, im in enumerate(ims)]
        rows.append(row)

    cntrl_row = [sg.Button('Previous Page', key='-PREV_PAGE-', enable_events=True),
                 sg.Button('Next Page', key='-NEXT_PAGE-', enable_events=True),
                 sg.Text(f'Current Page: {page + 1}', key='-CURR_PAGE_TEXT-')]
    rows.append(cntrl_row)
    return rows


def update_image_page(session, project, page, window, size):
    ims_added = 0
    offset = page * (size[0] * size[1])

    for y in range(size[0]):
        for x in range(size[1]):
            window[('-IMAGE-', (y, x))].update(image_filename=PLACEHOLDER_IM_PATH, image_size=(128, 64), image_subsample=2)

    for i in range(size[0]):
        ims = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (i * size[1])).limit(size[1]).all()
        ims_added += len(ims)
        for j, im in enumerate(ims):
            window[('-IMAGE-', (i, j))].update(image_filename=im.path, image_size=(128, 64), image_subsample=2)

    if ims_added > 0:
        window['-CURR_PAGE_TEXT-'].update(f'Current Page: {page + 1}')
        return True
    else:
        return False


def update_sel_image_browser(session, project, page, data, window, size):
    offset = page * (size[0] * size[1])
    im = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (data[0] * size[1])).limit(size[1]).all()[data[1]]
    window['-SEL_IMAGE-'].update(filename=im.path, size=(512, 256))

    y = pickle.loads(im.y)
    for i in range(len(y[0][0][0])):
        window[f'-CONTROL_LABEL_{i}-'].update(value=y[0][0][0][i])

    return im.id
    
    
def update_sel_image_child(window, im):
    window['-SEL_IMAGE-'].update(data=im, size=(512, 256))


def update_sel_image_timeline(session, project, page, data, window, size):
    offset = page * (size[0] * size[1])
    im = session.query(Image).filter_by(project=project).order_by(asc(Image.id)).offset(offset + (data[0] * size[1])).limit(size[1]).all()[data[1]]
    window['-SEL_IMAGE-'].update(filename=im.path, size=(512, 256))

    # y = pickle.loads(im.y)
    # for i in range(len(y[0][0])):
    #     window[f'-CONTROL_LABEL_{i}-'].update(value=y[0][0][i])

    return im.id


def create_children(session, gan, im_id, data):
    origin_im = session.query(Image).filter_by(id=im_id).first()

    session.query(Child).delete()
    session.commit()

    if origin_im is not None:
        z = np.array(pickle.loads(origin_im.z)).astype('float32')
        y = np.array(data[0]).astype('float32')
        z = z.reshape((1, 1, 1, gan.z_dim))
        y = y.reshape((1, 1, 1, gan.num_labels))

        rand = data[1]
        amount = int(data[2])

        for i in range(amount):
            z_mod = np.random.normal(size=[1, 1, 1, gan.z_dim]).astype('float32') * rand
            new_z = z + z_mod
            print(new_z)
            new_child = Child(z=pickle.dumps(new_z), y=pickle.dumps(y))
            session.add(new_child)
            session.commit()


def show_children(session, gan, window):
    children = session.query(Child).order_by(asc(Child.id)).all()
    
    base64_strs = []

    for y in range(IM_CHILDREN_SIZE[1]):
        for x in range(IM_CHILDREN_SIZE[0]):
            window[('-IMAGE_CHILDREN-', (x, y))].update(image_filename=PLACEHOLDER_IM_PATH, image_size=(128, 64), image_subsample=2)

    for i in range(len(children)):
        child = children[i]
        x = i % IM_CHILDREN_SIZE[0]
        y = i // IM_CHILDREN_SIZE[0]
        
        image = gan.generate_image(z=pickle.loads(child.z), y=pickle.loads(child.y))
        image = image * 127.5 + 127.5
        image = image.astype(np.uint8)
        image = pi.fromarray(image[0])
        with BytesIO() as output_bytes:
            im = pi.fromarray(skimage.img_as_ubyte(image))
            im.save(output_bytes, 'PNG')
            bytes_data = output_bytes.getvalue()
        base64_str = base64.b64encode(bytes_data)
        base64_strs.append(base64_str)
        window[('-IMAGE_CHILDREN-', (x, y))].update(image_data=base64_str, image_size=(128, 64), image_subsample=2)
    return base64_strs


def save_image(session, gan, project, data, path):
    index = data[1] * IM_CHILDREN_SIZE[0] + data[0]
    child = session.query(Child).order_by(asc(Child.id)).all()[index]
    print(pickle.loads(child.z))
    image = gan.generate_image(z=pickle.loads(child.z), y=pickle.loads(child.y))
    image = image * 127.5 + 127.5
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


def update_timeline(window, timelines, offset):
    tls = timelines[offset: offset + IM_GALLERY_SIZE_TIMELINE[1]]
    for i in range(IM_GALLERY_SIZE_TIMELINE[1]):
        window[('-TIMELINE_IMAGE-', i)].update(filename=PLACEHOLDER_IM_PATH, size=(128, 64), subsample=2)
        window[('-TIMELINE_ORDER-', i)].update(value='0')

    for i in range(len(tls)):
        window[('-TIMELINE_IMAGE-', i)].update(filename=tls[i].image.path, size=(128, 64), subsample=2)
        window[('-TIMELINE_ORDER-', i)].update(value=str(tls[i].order))


def cubic_spline_interp(points, step_count):
    def cubic_spline_interp1d(y):
        x = np.linspace(0., 1., len(y))
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.linspace(0., 1., step_count)
        return interpolate.splev(xnew, tck, der=0)
    if points.shape[0] < 4:
        raise ValueError('Too few points for cubic interpolation: need 4, got {}'.format(points.shape[0]))
    return np.apply_along_axis(cubic_spline_interp1d, 0, points)


def export_timeline(gan, timelines, frames, loop, interp, path):
    ims = [tl.image for tl in timelines]
    if loop:
        ims.append(ims[0])

    z_keys = []
    y_keys = []

    for im in ims:
        z_keys.append(pickle.loads(im.z))
        print(pickle.loads(im.z))
        y_keys.append(pickle.loads(im.y))
    
    z_keys = np.asarray(z_keys)
    y_keys = np.asarray(y_keys)

    if interp == TIMELINE_INTERP_LINEAR:
        z_seq = cubic_spline_interp(z_keys, frames)
        y_seq = cubic_spline_interp(y_keys, frames)
    else:
        return

    gen_ims = []
    for i in range(z_seq.shape[0]):
        z = z_seq[i].reshape(-1, 1, 1, gan.z_dim)
        y = y_seq[i].reshape(-1, 1, 1, gan.num_labels)
        gen_im = gan.generate_image(z, y)
        gen_im = gen_im * 127.5 + 127.5
        gen_im = np.uint8(gen_im)
        gen_ims.append(gen_im)

    gen_ims = np.asarray(gen_ims)

    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

    for i, im in enumerate(gen_ims):
        file_path = os.path.join(path, f'{i:09d}.png')
        tf.io.write_file(file_path, tf.image.encode_png(tf.cast(im[0], tf.uint8)))


def make_window1(session, project, gan, im_page):
    img_browser_row = get_image_page(session, project, im_page, IM_GALLERY_SIZE_BROWSER)
    img_control = []
    label_strings = gan.get_label_strings()
    for i in range(gan.num_labels):
        cntrl = [sg.Text(label_strings[i]), sg.Slider(range=(0.0, 1.0), resolution=0.0001,
                                                      orientation='horizontal', expand_x=True,
                                                      key=f'-CONTROL_LABEL_{i}-', enable_events=True)]
        img_control.append(cntrl)

    img_control.append([sg.Text('random:'), sg.Slider(range=(0.0, 1.0), resolution=0.0001,
                                                      orientation='horizontal', expand_x=True,
                                                      key='-CONTROL_RANDOM-', default_value=0.5,
                                                      enable_events=True)])
    img_control.append([sg.Text('children:'), sg.Slider(range=(0, 20), resolution=1,
                                                        orientation='horizontal', expand_x=True,
                                                        key='-CONTROL_CHILDREN-', default_value=10,
                                                        enable_events=True)])
    img_control.append([sg.Button('Create Children', key='-CREATE_CHILDREN-'), sg.Button('Save Child', key='-SAVE_CHILD-')])
    img_sel = [[sg.Text('Selected image:')], [sg.Image(key='-SEL_IMAGE-', size=(512, 256))]]

    img_children = [[sg.Text('Children:')]]
    for x in range(IM_CHILDREN_SIZE[0]):
        row = [sg.Button(key=('-IMAGE_CHILDREN-', (x, y)), enable_events=True, image_filename=PLACEHOLDER_IM_PATH,
                         image_size=(128, 64), image_subsample=2) for y in range(IM_CHILDREN_SIZE[1])]
        img_children.append(row)

    layout = [[sg.Column(img_sel), sg.Column(img_control, expand_x=True)],
              [sg.Column(img_browser_row), sg.Column(img_children)]]

    return sg.Window('Image Browser', layout, size=(1200, 1000), finalize=True)


def make_window2(session, project, im_page):
    img_sel = [[sg.Text('Selected image:')],
               [sg.Image(key='-SEL_IMAGE-', size=(512, 256))],
               [sg.Button('Add To Timeline', key='-ADD_TO_TIMELINE-', enable_events=True)]]
    img_browser_row = get_image_page(session, project, im_page, IM_GALLERY_SIZE_TIMELINE)
    timeline = []
    for i in range(IM_GALLERY_SIZE_TIMELINE[1]):
        item = sg.Column([[sg.Image(filename=PLACEHOLDER_IM_PATH, size=(128, 64), key=('-TIMELINE_IMAGE-', i), subsample=2)],
                          [sg.InputText(default_text='0', size=10, key=('-TIMELINE_ORDER-', i), enable_events=True)],
                          [sg.Button('Remove', key=('-REMOVE_FROM_TIMELINE-', i), enable_events=True)]])
        timeline.append(item)

    timeline_navigation = [sg.Button('Previous', key='-PREV_TIMELINE-', enable_events=True),
                         sg.Button('Update Order', key='-UPDATE_ORDER-', enable_events=True),
                         sg.Button('Next', key='-NEXT_TIMELINE-', enable_events=True)]

    timeline_controls = [[sg.Text('Interpolation:'), sg.Combo(['linear', 'sine'], default_value='linear', key='-TIMELINE_INTERP-', enable_events=True)],
                         [sg.Text('Frames:'), sg.InputText(default_text='1000', key='-TIME_LINE_FRAMES-', enable_events=True)],
                         [sg.Text('Loop:'), sg.Checkbox('', default=True, key='-TIMELINE_LOOP-', enable_events=True)],
                         [sg.Button('Export', key='-TIMELINE_EXPORT-', enable_events=True)]]

    layout = [[sg.Column(img_sel), sg.Column(timeline_controls)], [timeline], [timeline_navigation], [sg.Column(img_browser_row)]]

    return sg.Window('Timeline', layout, size=(1200, 1000), finalize=True)


def main():
    parser = argparse.ArgumentParser(description='Latent Space GUI')
    parser.add_argument('-p', '--project_file', type=str, help='the path to the project file')

    args = parser.parse_args()
    if not os.path.isfile(args.project_file):
        print('Not a valid project path!')
        sys.exit()

    with open(args.project_file) as f:
        conf = json.load(f)

    gan = dcgan.DCGAN(conf)
    gen_img_path = os.path.join(conf['images_path'], 'generated')
    timeline_img_path = os.path.join(conf['images_path'], 'timeline')

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
    control_data = [[[[0. for _ in range(gan.num_labels)]]], 0.5, 10]
    im_sel_child = None
    children_ims = []

    project = session.query(Project).filter_by(path=conf['project_path']).first()
    if project is None:
        project = Project(path=conf['project_path'])
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
    window2, window1= make_window2(session, project, im_page_timeline), make_window1(session, project, gan, im_page_browser)

    update_timeline(window2, timelines, timeline_offset)

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
                    im_sel_id_browser = update_sel_image_browser(session, project, im_page_browser, event[1], window, IM_GALLERY_SIZE_BROWSER)
                elif window == window2:
                    im_sel_id_timeline = update_sel_image_timeline(session, project, im_page_timeline, event[1], window, IM_GALLERY_SIZE_TIMELINE)

            if event[0] == '-IMAGE_CHILDREN-':
                im_sel_child = event[1]
                print(im_sel_child)
                print(im_sel_child[1] * IM_CHILDREN_SIZE[0] + im_sel_child[0])
                update_sel_image_child(window, children_ims[im_sel_child[1] * IM_CHILDREN_SIZE[0] + im_sel_child[0]])

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
                update_timeline(window, timelines, timeline_offset)
                
        else:
            if event == '-NEXT_PAGE-':
                if window == window1:
                    new_page = im_page_browser + 1
                    if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_BROWSER):
                        im_page_browser = new_page
                elif window == window2:
                    new_page = im_page_timeline + 1
                    if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_TIMELINE):
                        im_page_timeline = new_page

            if event == '-PREV_PAGE-':
                if window == window1:
                    new_page = im_page_browser - 1
                    if im_page_browser <= 0:
                        im_page_browser = 0
                    else:
                        if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_BROWSER):
                            im_page_browser = new_page
                elif window == window2:
                    new_page = im_page_timeline - 1
                    if im_page_timeline <= 0:
                        im_page_timeline = 0
                    else:
                        if update_image_page(session, project, new_page, window, IM_GALLERY_SIZE_TIMELINE):
                            im_page_timeline = new_page

            if event == '-CONTROL_RANDOM-':
                control_data[1] = values['-CONTROL_RANDOM-']

            if event == '-CONTROL_CHILDREN-':
                control_data[2] = values['-CONTROL_CHILDREN-']

            if event.startswith('-CONTROL_LABEL_'):
                data = values[event]
                index = int(event.replace('-CONTROL_LABEL_', '').replace('-', ''))
                control_data[0][0][0][index] = data

            if event == '-CREATE_CHILDREN-':
                create_children(session, gan, im_sel_id_browser, control_data)
                children_ims = show_children(session, gan, window)

            if event == '-SAVE_CHILD-':
                save_image(session, gan, project, im_sel_child, gen_img_path)

            if event == '-ADD_TO_TIMELINE-':
                timelines = add_image_to_timeline(session, project, im_sel_id_timeline, im_sel_id_timeline_pos)
                im_sel_id_timeline_pos += 1
                update_timeline(window, timelines, timeline_offset)
                
            #if event == '-REMOVE_FROM_TIMELINE-':
            #   print('remove from timeline')

            if event == '-PREV_TIMELINE-':
                if not timeline_offset <= 0:
                    timeline_offset -= 1
                    update_timeline(window, timelines, timeline_offset)

            if event == '-NEXT_TIMELINE-':
                timeline_offset += 1
                update_timeline(window, timelines, timeline_offset)

            if event == '-UPDATE_ORDER-':
                timelines = session.query(Timeline).filter_by(project=project).order_by(asc(Timeline.order)).all()
                update_timeline(window, timelines, timeline_offset)

            if event == '-TIMELINE_INTERP-':
                timeline_interp = values[event]

            if event == '-TIME_LINE_FRAMES-':
                timeline_frames = int(values[event])

            if event == '-TIMELINE_LOOP-':
                timeline_loop = values[event]

            if event == '-TIMELINE_EXPORT-':
                export_timeline(gan, timelines, timeline_frames, timeline_loop, timeline_interp, timeline_img_path)


if __name__ == '__main__':
    main()
