"""
module providing napari widget
"""

import logging
import numpy as np
import torch
import warnings

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

from lsfm_destripe_napari._reader import open_dialog, napari_get_reader
from lsfm_destripe_napari._writer import save_dialog, write_tiff



class DestripeWidget(QWidget):
    """Main widget of the plugin"""


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.advanced_options_elements = []
        
        # This is how to set position and size of the viewer window:
        # self.viewer.window.set_geometry(0, 0, max(1000, width), max(600, height))

        _, _, width, height = self.viewer.window.geometry()
        # width = self.viewer.window.geometry()[2]
        # height = self.viewer.window.geometry()[3]
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

        label_resample = QLabel("Resample ratio:")
        label_kernel_size = QLabel("GF Kernel size inference:")
        label_kernel_size.setToolTip("Must be odd")
        label_lambda_hessian = QLabel("Lambda Hessian:")
        label_hessian_kernel_sigma = QLabel("Hessian Kernel Sigma:")
        label_isotropic_hessian = QLabel("Isotropic Hessian:")

        self.advanced_options_elements.extend([
            label_resample,
            label_kernel_size,
            label_lambda_hessian,
            label_hessian_kernel_sigma,
            label_isotropic_hessian,
        ])

        # QPushbutton
        btn_load = QPushButton("Load")
        btn_process = QPushButton("Process")
        btn_save = QPushButton("Save")
        self.btn_advanced_options = QPushButton("Hide Advanced Options")

        btn_load.clicked.connect(self.load)
        btn_process.clicked.connect(self.process)
        btn_save.clicked.connect(self.save)
        self.btn_advanced_options.clicked.connect(self.toggle_advanced_options)

        # QCombobox
        self.combobox_mask = QComboBox()

        # QCheckBox
        self.checkbox_vertical = QCheckBox()
        self.checkbox_vertical.setChecked(True)
        self.checkbox_iso_hessian = QCheckBox()
        self.checkbox_iso_hessian.setChecked(True)

        self.advanced_options_elements.append(self.checkbox_iso_hessian)

        # QLineEdit
        self.lineedit_angle = QLineEdit()
        self.lineedit_angle.setText("0")
        self.lineedit_resample = QLineEdit()
        self.lineedit_resample.setText("3")
        self.lineedit_kernel_size = QLineEdit()
        self.lineedit_kernel_size.setText("29")
        self.lineedit_lambda_hessian = QLineEdit()
        self.lineedit_lambda_hessian.setText("1")
        self.lineedit_hessian_kernel_sigma = QLineEdit()
        self.lineedit_hessian_kernel_sigma.setText("1")

        self.advanced_options_elements.extend([
            self.lineedit_resample,
            self.lineedit_kernel_size,
            self.lineedit_lambda_hessian,
            self.lineedit_hessian_kernel_sigma,
        ])

        # QGroupBox
        parameters = QGroupBox("Parameters")
        gb_layout = QGridLayout()
        gb_layout.addWidget(label_vertical, 0, 0)
        gb_layout.addWidget(self.checkbox_vertical, 0, 1)
        gb_layout.addWidget(label_angle, 1, 0)
        gb_layout.addWidget(self.lineedit_angle, 1, 1)
        gb_layout.addWidget(label_mask, 2, 0)
        gb_layout.addWidget(self.combobox_mask, 2, 1)
        gb_layout.addWidget(self.btn_advanced_options, 3, 0, 1, -1)
        gb_layout.addWidget(label_resample, 4, 0)
        gb_layout.addWidget(self.lineedit_resample, 4, 1)
        gb_layout.addWidget(label_kernel_size, 5, 0)
        gb_layout.addWidget(self.lineedit_kernel_size, 5, 1)
        gb_layout.addWidget(label_lambda_hessian, 6, 0)
        gb_layout.addWidget(self.lineedit_lambda_hessian, 6, 1)
        gb_layout.addWidget(label_hessian_kernel_sigma, 7, 0)
        gb_layout.addWidget(self.lineedit_hessian_kernel_sigma, 7, 1)
        gb_layout.addWidget(label_isotropic_hessian, 8, 0)
        gb_layout.addWidget(self.checkbox_iso_hessian, 8, 1)
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

        self.toggle_advanced_options()
        
        def wrapper(self, func, event):
            self.logger.debug("Exiting...")
            return func(event)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func = self.viewer.window._qt_window.closeEvent
            self.viewer.window._qt_window.closeEvent = lambda event: wrapper(self, func, event)

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
        params = self.get_parameters()
        if params is None:
            return
        output_image = DeStripe.train_on_full_arr(
            X = params["input_image"],
            is_vertical = params["is_vertical"],
            angle_offset = params["angle_offset"],
            mask = params["mask"],
            device = params["device"],
        )
        self.viewer.add_image(output_image, name="Destriped Image")

    def get_parameters(self):
        mask_layer_name = self.combobox_mask.currentText()
        if mask_layer_name not in self.viewer.layers:
            self.logger.info("Selected mask not found")
            return
        params = {}
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                params["input_image"] = layer.data
                self.logger.debug(f"Using layer: {layer.name}")
            break
        if params["input_image"] is None:
            self.logger.info("No image layer found")
            return
        params["input_image"] = np.expand_dims(params["input_image"], axis=1)
        self.logger.debug(f"Selected mask: {mask_layer_name}")
        params["is_vertical"] = self.checkbox_vertical.isChecked()
        self.logger.debug(f"Vertical: {params['is_vertical']}")
        try:
            params["angle_offset"] = list(map(float, self.lineedit_angle.text().split(",")))
        except ValueError:
            self.logger.error("Invalid angle offset")
            return
        self.logger.debug(f"Angle offset: {params['angle_offset']}")
        mask_layer_index = self.viewer.layers.index(mask_layer_name)
        params["mask"] = self.viewer.layers[mask_layer_index].data
        params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"Device: {params['device']}")
        try:
            params["resample_ratio"] = int(self.lineedit_resample.text())
        except ValueError:
            self.logger.error("Invalid resample ratio")
            return
        if not params["resample_ratio"] in range(1, 6):
            self.logger.error("Resample ratio must be between 1 and 5")
            return
        self.logger.debug(f"Resample ratio: {params['resample_ratio']}")
        try:
            params["gf_kernel_size_inference"] = int(self.lineedit_kernel_size.text())
        except ValueError:
            self.logger.error("Invalid GF kernel size inference")
            return
        if params["gf_kernel_size_inference"] not in range(29, 90, 2):
            self.logger.error("GF kernel size inference must be odd and between 29 and 89")
            return
        self.logger.debug(f"GF kernel size inference: {params['gf_kernel_size_inference']}")
        try:
            params["lambda_hessian"] = float(self.lineedit_lambda_hessian.text())
        except ValueError:
            self.logger.error("Invalid lambda Hessian")
            return
        if not (0 <= params["lambda_hessian"] and params["lambda_hessian"] <= 10):
            self.logger.error("Lambda Hessian must be between 0 and 10")
            return
        self.logger.debug(f"Lambda Hessian: {params['lambda_hessian']}")
        try:
            params["hessian_kernel_sigma"] = float(self.lineedit_hessian_kernel_sigma.text())
        except ValueError:
            self.logger.error("Invalid Hessian kernel sigma")
            return
        if not (0.5 <= params["hessian_kernel_sigma"] and params["hessian_kernel_sigma"] <= 1.5):
            self.logger.error("Hessian kernel sigma must be between 0.5 and 1.5")
            return
        self.logger.debug(f"Hessian kernel sigma: {params['hessian_kernel_sigma']}")
        params["isotropic_hessian"] = self.checkbox_iso_hessian.isChecked()
        self.logger.debug(f"Isotropic Hessian: {params['isotropic_hessian']}")
        return params

    def toggle_advanced_options(self):
        if self.btn_advanced_options.text() == "Show Advanced Options":
            self.btn_advanced_options.setText("Hide Advanced Options")
            show = True
        else:
            self.btn_advanced_options.setText("Show Advanced Options")
            show = False
        for element in self.advanced_options_elements:
            element.setVisible(show)

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
