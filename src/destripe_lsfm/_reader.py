"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""

from pathlib import Path
from qtpy.QtWidgets import QFileDialog
from aicsimageio import AICSImage


def open_dialog(parent):
    """
    Opens a dialog to select a file to open

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog

    Returns
    -------
    str
        Path of the selected file
    """
    dialog = QFileDialog()
    filepath, _ = dialog.getOpenFileName(
        parent, "Select Image-File", filter="TIFF files (*.tif *.tiff)"
    )
    return filepath


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str
        Path to file

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # if isinstance(path, list):
    #     # reader plugins may be handed single path, or a list of paths.
    #     # if it is a list, it is assumed to be an image stack...
    #     # so we are only going to look at the first file.
    #     path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not (path.endswith(".tif") or path.endswith(".tiff")):
        return None

    # otherwise we return the *function* that can read ``path``.
    return read_tiff


def read_tiff(path):
    """
    Read a TIFF file

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    np.ndarray
        Data read from the file
    """
    filename = Path(path).name
    reader = AICSImage(path)
    data = reader.get_image_data("ZYX")
    return data, filename
