import os
from sqlalchemy import Column, ForeignKey, LargeBinary, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import pickle

Base = declarative_base()

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui.db')


class Project(Base):
    __tablename__ = 'project'

    id = Column(Integer, primary_key=True)
    path = Column(String(400), nullable=False)

    def __repr__(self):
        return f'Project | id: {self.id}, path: {self.path}'


class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)
    path = Column(String(400), nullable=False)
    z = Column(LargeBinary, nullable=False)
    y = Column(LargeBinary, nullable=False)
    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(Project)

    def __repr__(self):
        return f'Image | id: {self.id}, z: {pickle.loads(self.z)}, y: {pickle.loads(self.y)}, project: {self.project}'


class Child(Base):
    __tablename__ = 'child'

    id = Column(Integer, primary_key=True)
    z = Column(LargeBinary, nullable=False)
    y = Column(LargeBinary, nullable=False)

    def __repr__(self):
        return f'Image | id: {self.id}, z: {pickle.loads(self.z)}, y: {pickle.loads(self.y)}'


class Timeline(Base):
    __tablename__ = 'timeline'

    id = Column(Integer, primary_key=True)
    order = Column(Integer, nullable=True)
    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(Project)
    image_id = Column(Integer, ForeignKey('image.id'))
    image = relationship(Image)

    def __repr__(self):
        return f'Timeline | id: {self.id}, order: {self.order}, project: {self.project}, image: {self.image}'


engine = create_engine(f'sqlite:///{path}')
Base.metadata.create_all(engine)
