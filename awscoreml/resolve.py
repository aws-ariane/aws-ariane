import os
import types

import awscoreml.data as location


class resource(object):
    @staticmethod
    def filename(package, filename):
        if isinstance(package, types.ModuleType):
            parts = filename.split('/')
            parts.insert(0, os.path.dirname(package.__file__))
        return os.path.join(*parts)

    @staticmethod
    def exists(package, filename):
        return os.path.exists(resource.filename(package, filename))


class sagemaker(object):

    @staticmethod
    def check():
        path = os.path.join(*[os.sep, 'opt', 'ml'])
        return os.path.exists(path)

    @staticmethod
    def model(filename):
        return os.path.join(*[os.sep, 'opt', 'ml', 'model', filename])

    @staticmethod
    def input(channel, filename):
        return os.path.join(*[os.sep, 'opt', 'ml', 'input', 'data', channel, filename])

    @staticmethod
    def config(filename):
        return os.path.join(*[os.sep, 'opt', 'ml', 'input', 'config', filename])

    @staticmethod
    def failure():
        return os.path.join(*[os.sep, 'opt', 'ml', 'output', 'failure'])

    @staticmethod
    def output(filename):
        return os.path.join(*[os.sep, 'opt', 'ml', 'model', filename])


class local(object):

    @staticmethod
    def model(filename):
        return resource.filename(location, filename)

    @staticmethod
    def input(channel, filename):
        return resource.filename(location, filename)

    @staticmethod
    def config(filename):
        return resource.filename(location, filename)

    @staticmethod
    def failure():
        return resource.filename(location, 'failure')

    @staticmethod
    def output(filename):
        return resource.filename(location, filename)


class paths(object):

    @staticmethod
    def base():
        if sagemaker.check():
            return sagemaker
        return local

    @staticmethod
    def model(filename):
        return paths.base().model(filename)

    @staticmethod
    def input(channel, filename):
        return paths.base().input(channel, filename)

    @staticmethod
    def config(filename):
        return paths.base().config(filename)

    @staticmethod
    def failure():
        return paths.base().failure()

    @staticmethod
    def output(filename):
        return paths.base().output(filename)
