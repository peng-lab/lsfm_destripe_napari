"""
module providing napari widget
"""

import logging
import numpy as np
import torch

from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QWidget,
    QScrollArea,
    QGridLayout,
    QLabel,
    QGroupBox,
    QComboBox,
    QDialog,
    QApplication,
    QCheckBox,
    QLineEdit,
)
from qtpy.QtCore import Qt
import napari

from lsfm_destripe.core import DeStripe

from destripe_lsfm._reader import open_dialog, napari_get_reader
from destripe_lsfm._writer import save_dialog, write_tiff



class DestripeWidget(QWidget):
    """Main widget of the plugin"""

    print(callable(DeStripe)) # sanity check for vscode

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        
        # This is how to set position and size of the viewer window:
        # self.viewer.window.set_geometry(0, 0, max(1000, width), max(600, height))

        _, _, width, height = self.viewer.window.geometry()
        width = self.viewer.window.geometry()[2]
        height = self.viewer.window.geometry()[3]
        self.viewer.window.resize(max(1000, width),max(600, height))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing DestripeWidget...")

        ## QObjects
        # QLabel
        title = QLabel("<h1>LSFM DeStripe</h1>")
        title.setAlignment(Qt.AlignCenter)
        title.setMaximumHeight(100)

        label_mask = QLabel("Mask:")
        label_vertical = QLabel("Is vertical:")
        label_angle = QLabel("Angle offset:")
        label_angle.setToolTip("Angle offset in degrees")

        # QPushbutton
        btn_load = QPushButton("Load")
        btn_process = QPushButton("Process")
        btn_save = QPushButton("Save")

        btn_load.clicked.connect(self.load)
        btn_process.clicked.connect(self.process)
        btn_save.clicked.connect(self.save)

        # QCombobox
        self.combobox_mask = QComboBox()
        # self.combobox_mask.addItems(
        #     ["Layernames", "will", "show", "up", "here"]
        # )

        # QCheckBox
        self.checkbox_vertical = QCheckBox()
        self.checkbox_vertical.setChecked(True)

        # QLineEdit
        self.lineedit_angle = QLineEdit()
        self.lineedit_angle.setText("0")

        # QGroupBox
        parameters = QGroupBox("Parameters")
        gb_layout = QGridLayout()
        gb_layout.addWidget(label_vertical, 0, 0)
        gb_layout.addWidget(self.checkbox_vertical, 0, 1)
        gb_layout.addWidget(label_angle, 1, 0)
        gb_layout.addWidget(self.lineedit_angle, 1, 1)
        gb_layout.addWidget(label_mask, 2, 0)
        gb_layout.addWidget(self.combobox_mask, 2, 1)
        parameters.setLayout(gb_layout)

        layout = QGridLayout()
        layout.addWidget(title, 0, 0, 1, -1)
        layout.addWidget(btn_load, 1, 0)
        layout.addWidget(parameters, 2, 0, 1, -1)
        layout.addWidget(btn_process, 3, 0)
        layout.addWidget(btn_save, 3, 1)

        widget = QWidget()
        widget.setLayout(layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)
        self.setMinimumWidth(300)
        QApplication.instance().aboutToQuit.connect(lambda: self.logger.debug("Exiting..."))

        self.viewer.layers.events.inserted.connect(self.update_combobox)
        self.viewer.layers.events.inserted.connect(self.connect_rename)
        self.viewer.layers.events.removed.connect(self.update_combobox)
        self.viewer.layers.events.reordered.connect(self.update_combobox)
        for layer in self.viewer.layers:
            layer.events.name.connect(self.update_combobox)
        self.update_combobox()

        self.logger.debug("DestripeWidget initialized")
        self.logger.info("Ready to use")

    def update_combobox(self):
        self.logger.debug("Updating combobox...")
        layernames = [
            layer.name
            for layer in self.viewer.layers
            if type(layer) == napari.layers.Labels
        ]
        layernames.reverse()
        self.combobox_mask.clear()
        self.combobox_mask.addItems(layernames)

    def connect_rename(self, event):
        event.value.events.name.connect(self.update_combobox)

    def load(self):
        self.logger.info("Waiting for user to select a file...")
        filepath = open_dialog(self)
        self.logger.debug("Getting reader for file...")
        reader = napari_get_reader(filepath)
        if reader is None:
            self.logger.info("No reader found for file")
            return
        self.logger.debug("Reading file...")
        image, filename = reader(filepath)
        self.logger.debug(f"Image shape: {image.shape}")
        self.logger.debug(f"Image dtype: {image.dtype}")
        self.logger.debug(f"Adding image to viewer as {filename}...")
        self.viewer.add_image(image, name=filename)
        self.logger.info("Image added to viewer")

    def save(self):
        layernames = [
            layer.name
            for layer in self.viewer.layers
            if type(layer) == napari.layers.Image
        ]
        layernames.reverse()
        if not layernames:
            self.logger.info("No image layers found")
            return
        if len(layernames) == 1:
            self.logger.info("Only one image layer found")
            layername = layernames[0]
        else:
            self.logger.info("Multiple image layers found")
            dialog = LayerSelection(layernames)
            index = dialog.exec_()
            if index == -1:
                self.logger.info("No layer selected")
                return
            layername = layernames[index]
        self.logger.debug(f"Selected layer: {layername}")
        data = self.viewer.layers[self.viewer.layers.index(layername)].data
        self.logger.debug(f"Data shape: {data.shape}")
        self.logger.debug(f"Data dtype: {data.dtype}")
        filepath = save_dialog(self)
        if filepath == ".tiff" or filepath == ".tif":
            self.logger.info("No file selected")
            return
        self.logger.debug(f"Saving to {filepath}...")
        write_tiff(filepath, data)
        self.logger.info("Data saved")

    def process(self):
        mask_layer_name = self.combobox_mask.currentText()
        self.logger.debug("Selected mask: %s", mask_layer_name)
        if mask_layer_name not in self.viewer.layers:
            self.logger.info("Selected mask not found")
            return
        is_vertical = self.checkbox_vertical.isChecked()
        self.logger.debug("Vertical: %s", is_vertical)
        try:
            angle_offset = list(map(float, self.lineedit_angle.text().split(",")))
        except ValueError:
            self.logger.error("Invalid angle offset")
            return
        self.logger.debug("Angle offset: %s", angle_offset)
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                input_image = layer.data
            break
        if input_image is None:
            self.logger.info("No image layer found")
            return
        input_image = np.expand_dims(input_image, axis=1)
        mask_layer_index = self.viewer.layers.index(mask_layer_name)
        mask_arr = self.viewer.layers[mask_layer_index].data
        if torch.cuda.is_available():
            self.logger.debug("CUDA is available")
            device = "cuda"
        else:
            self.logger.debug("CUDA is not available")
            device = "cpu"
        output_image = DeStripe.train_on_full_arr(
            X = input_image,
            is_vertical = is_vertical,
            angle_offset = angle_offset,
            mask = mask_arr,
            device = device,
        )
        self.viewer.add_image(output_image, name="Destriped Image")

class LayerSelection(QDialog):
    def __init__(self, layernames: list[str]):
        super().__init__()
        self.setWindowTitle("Select Layer to save as TIFF")
        self.combobox = QComboBox()
        self.combobox.addItems(layernames)
        btn_select = QPushButton("Select")
        btn_select.clicked.connect(self.accept)
        layout = QVBoxLayout()
        layout.addWidget(self.combobox)
        layout.addWidget(btn_select)
        self.setLayout(layout)
        self.setMinimumSize(250, 100)

    def accept(self):
        self.done(self.combobox.currentIndex())

    def reject(self):
        self.done(-1)
