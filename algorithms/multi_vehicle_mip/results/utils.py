import os

from common.custom_types import OutputVideoName, OutputVideoPath


def get_full_path_of_output_video(
    output_video_filename: OutputVideoName,
) -> OutputVideoPath:
    directory_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(directory_path, output_video_filename)
