from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lygia',
    maintainer_email='lygia@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    scripts=['scripts/detector_node'],
)
