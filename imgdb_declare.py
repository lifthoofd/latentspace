import os
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import numpy as np
import pickle

Base = declarative_base()

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database/images.db')


class Project(Base):
    __tablename__ = 'project'

    id = Column(Integer, primary_key=True)

    name = Column(String(100), nullable=False)
    path = Column(String(400), nullable=False)

    def __repr__(self):
        return 'Project | name: {}, path: {}'.format(self.name, self.path)

    def get_dict(self):
        return {'id': self.id,
                'name': self.name,
                'path': self.path}


class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)

    z = Column(LargeBinary, nullable=False)
    y = Column(LargeBinary, nullable=False)

    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(Project)

    def __repr__(self):
        return 'Image | id: {}, z: {}, y: {}'.format(self.id, pickle.loads(self.z), pickle.loads(self.y))

    def get_dict(self):
        return {'id': self.id,
                'z': pickle.loads(self.z),
                'y': pickle.loads(self.y)}


class Timeline(Base):
    __tablename__ = 'timeline'

    id = Column(Integer, primary_key=True)

    order = Column(Integer, nullable=True)

    image_id = Column(Integer, ForeignKey('image.id'))
    image = relationship(Image)

    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(Project)

    def get_dict(self):
        return {'id': self.id,
                'order': self.order,
                'image': self.image.get_dict(),
                'project': self.project.get_dict()}


# create engine
engine = create_engine('sqlite:///{}'.format(path))

# create all tables
Base.metadata.create_all(engine)