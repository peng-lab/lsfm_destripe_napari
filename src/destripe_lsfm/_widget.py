"""
module providing napari widget
"""

import logging

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
)
from qtpy.QtCore import Qt
import napari

from destripe_lsfm._reader import open_dialog, napari_get_reader
from destripe_lsfm._writer import save_dialog, write_tiff



class DestripeWidget(QWidget):
    """Main widget of the plugin"""

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
        label_p1 = QLabel("P1:")
        label_p2 = QLabel("P2:")

        # QPushbutton
        btn_load = QPushButton("Load")
        btn_process = QPushButton("Process")
        btn_save = QPushButton("Save")

        btn_load.clicked.connect(self.load)
        btn_process.clicked.connect(self.process)
        btn_save.clicked.connect(self.save)

        # QCombobox
        self.combobox_input_layer = QComboBox()
        # self.combobox_input_layer.addItems(
        #     ["Layernames", "will", "show", "up", "here"]
        # )

        # QGroupBox
        parameters = QGroupBox("Parameters")
        gb_layout = QGridLayout()
        gb_layout.addWidget(label_p1, 0, 0)
        gb_layout.addWidget(label_p2, 1, 0)
        gb_layout.addWidget(label_mask, 2, 0)
        gb_layout.addWidget(self.combobox_input_layer, 2, 1)
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
            if type(layer) == napari.layers.Image
        ]
        layernames.reverse()
        self.combobox_input_layer.clear()
        self.combobox_input_layer.addItems(layernames)

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
        # data = self.viewer.layers[0].data
        write_tiff(filepath, data)
        self.logger.info("Data saved")

    def process(self):
        self.logger.info("Processing coming soon...")


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
