import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService

def generate_launch_description():
    detector_node = Node(
        package='detector',
        executable='detector_node',
        name='detector_node',
        output='screen',
        parameters=[os.path.join(get_package_share_directory("detector"), 'params', 'detector.yaml')],

    )

    ld = LaunchDescription()

    ld.add_action(detector_node)

    return ld

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()