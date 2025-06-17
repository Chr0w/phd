from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

map_path = os.path.join(
    get_package_share_directory('cpp_map_publisher'),
    'maps',
    'm1.png'
)


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cpp_map_publisher',
            namespace='',
            executable='map_publisher',
            name='map_publisher',
            parameters=[{'map_image': map_path}],
        ),
        Node(
            package='edi_calculator',
            namespace='',
            executable='edi_calculator_node',
            name='edi_calculator',
        ),
    ])