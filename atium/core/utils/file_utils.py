import inspect
import os

from atium.core.utils.custom_types import FileName, FilePath

RESULTS_DIR_NAME = "results"
IMPLEMENTATIONS_DIR_NAME = "implementations"
EXPERIMENTS_DIR_NAME = "experiments"


def get_file_path_in_implementations_results_dir(
    output_filename: FileName,
) -> FilePath:
    # We can't use __file__ as it gives us the path to the current file in common/
    # Using inspect to get the directory of the calling file which is then used to
    # recover the full path to the results of the calling algorithm.
    directory_path = os.path.dirname(os.path.realpath(inspect.stack()[1].filename))
    assert os.path.exists(directory_path), f"Directory {directory_path} does not exist!"
    assert (
        IMPLEMENTATIONS_DIR_NAME in directory_path
    ), f"Results must be stored under {IMPLEMENTATIONS_DIR_NAME}/**/{RESULTS_DIR_NAME}/"
    return os.path.join(directory_path, RESULTS_DIR_NAME, output_filename)
