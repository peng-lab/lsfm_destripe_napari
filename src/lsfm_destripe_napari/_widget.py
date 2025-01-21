"""
module providing napari widget
"""

import logging
import numpy as np
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
    QCheckBox,
    QLineEdit,
    QStylePainter,
    QStyleOptionComboBox,
    QStyle,
    QSlider,
)
from qtpy import QtGui
from qtpy.QtCore import Qt
import napari

from lsfm_destripe.core import DeStripe

from lsfm_destripe_napari._reader import open_dialog, napari_get_reader
from lsfm_destripe_napari._writer import save_dialog, write_tiff

font = QtGui.QFont()
font.setPointSize(20)


class ComboBox(QComboBox):
    # https://code.qt.io/cgit/qt/qtbase.git/tree/src/widgets/widgets/qcombobox.cpp?h=5.15.2#n3173
    def paintEvent(self, event):

        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))

        # draw the combobox frame, focusrect and selected etc.
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)

        if self.currentIndex() < 0:
            opt.palette.setBrush(
                QtGui.QPalette.ButtonText,
                opt.palette.brush(QtGui.QPalette.ButtonText).color().lighter(),
            )
            if self.placeholderText():
                opt.currentText = self.placeholderText()

        # draw the icon and text
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)


class DestripeWidget(QWidget):
    """Main widget of the plugin"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.advanced_options_elements = []

        # This is how to set position and size of the viewer window:
        # self.viewer.window.set_geometry(0, 0, max(1000, width),
        # max(600, height))

        _, _, width, height = self.viewer.window.geometry()
        # width = self.viewer.window.geometry()[2]
        # height = self.viewer.window.geometry()[3]
        self.viewer.window.resize(max(1000, width), max(600, height))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing DestripeWidget...")

        # QLabel
        title = QLabel("<h1>Leonardo-DeStripe</h1>")
        title.setAlignment(Qt.AlignCenter)
        title.setMaximumHeight(100)

        label_mask = QLabel("Mask (label layer):")
        label_vertical = QLabel("Is vertical:")
        label_angle = QLabel("Angle offset(s):")
        label_backend = QLabel("Backend: ")
        label_angle.setToolTip("Angle offset in degrees")

        # label_resample = QLabel("Resample ratio:")
        # label_kernel_size = QLabel("GF Kernel size inference:")
        # label_kernel_size.setToolTip("Must be odd")
        self.label_lambda = QLabel("Lambda")
        self.label_lambda_hessian = QLabel("Hessian: ")
        self.label_lambda_tv = QLabel("TV: ")
        self.label_lambda_mse = QLabel("MSE: ")
        self.label_hessian_kernel_sigma = QLabel("Hessian kernel sigma: 1.00")
        self.label_downsample_ratio = QLabel("Downsample ratio: 3")
        self.label_n_epochs = QLabel("Training epochs: ")
        self.label_wedge = QLabel("Angular size of mask: 29")
        self.label_inc = QLabel("Latent dimension: ")
        self.label_neighbors = QLabel("GNN neighbors: ")
        self.label_guided_upsample_kernel_length = QLabel(
            "Upsample kernel: 49"
        )
        self.label_dark_stripe_only = QLabel("Dark stripe only: ")

        self.advanced_options_elements.extend(
            [
                self.label_downsample_ratio,
                self.label_hessian_kernel_sigma,
                self.label_lambda_mse,
                self.label_lambda_hessian,
                self.label_lambda_tv,
                self.label_lambda,
                self.label_n_epochs,
                self.label_wedge,
                self.label_inc,
                self.label_neighbors,
            ]
        )

        # QPushbutton
        # btn_load = QPushButton("Load")
        self.btn_load = ComboBox()
        self.btn_load.setPlaceholderText("--Select input from image layers--")
        self.btn_load.setCurrentIndex(-1)
        self.btn_load.currentIndexChanged.connect(self.toggle_input_display)
        btn_process = QPushButton("Process")
        btn_save = QPushButton("Save")
        self.btn_advanced_options = QPushButton("Hide Advanced Parameters")
        self.angle_check_button = QPushButton("check")

        # btn_load.clicked.connect(self.load)
        btn_process.clicked.connect(self.process)
        btn_save.clicked.connect(self.save)
        self.btn_advanced_options.clicked.connect(self.toggle_advanced_options)
        self.angle_check_button.clicked.connect(self.toggle_check_angle)

        # QCombobox
        self.combobox_mask = QComboBox()
        self.combobox_baskend = QComboBox()
        self.combobox_baskend.addItem("jax")
        self.combobox_baskend.addItem("torch")
        self.backend = "jax"
        self.combobox_baskend.currentIndexChanged.connect(self.backend_changed)

        # QCheckBox
        self.checkbox_vertical = QCheckBox()
        self.checkbox_vertical.setChecked(True)
        self.checkbox_dark_stripe_only = QCheckBox()
        self.checkbox_dark_stripe_only.setChecked(False)

        # QSlider
        sld_downsample = QSlider(Qt.Horizontal)
        sld_downsample.setRange(2, 5)
        sld_downsample.setValue(3)
        self.lineedit_downsample = 3
        sld_downsample.valueChanged.connect(self.downsample)

        sld_hessian_sigma = QSlider(Qt.Horizontal)
        sld_hessian_sigma.setRange(0, 5)
        sld_hessian_sigma.setValue(2)
        self.lineedit_hessian_sigma = 1
        sld_hessian_sigma.valueChanged.connect(self.hessian_sigma)

        sld_wedge = QSlider(Qt.Horizontal)
        sld_wedge.setRange(1, 90)
        sld_wedge.setValue(29)
        self.lineedit_wedge = 29
        sld_wedge.valueChanged.connect(self.wedge)

        sld_guided_upsample_kernel_length = QSlider(Qt.Horizontal)
        sld_guided_upsample_kernel_length.setRange(0, 95)
        sld_guided_upsample_kernel_length.setValue(20)
        self.lineedit_guided_upsample_kernel_length = 49
        sld_guided_upsample_kernel_length.valueChanged.connect(
            self.guided_upsample_kernel_length
        )

        # QLineEdit
        self.lineedit_angle = QLineEdit()
        self.lineedit_angle.setText("0")
        self.lineedit_lambda_hessian = QLineEdit()
        self.lineedit_lambda_hessian.setText("1")
        self.lineedit_lambda_tv = QLineEdit()
        self.lineedit_lambda_tv.setText("1")
        self.lineedit_lambda_mse = QLineEdit()
        self.lineedit_lambda_mse.setText("1")
        self.lineedit_n_epochs = QLineEdit()
        self.lineedit_n_epochs.setText("300")
        self.lineedit_inc = QLineEdit()
        self.lineedit_inc.setText("16")
        self.lineedit_neighbors = QLineEdit()
        self.lineedit_neighbors.setText("16")

        self.advanced_options_elements = self.advanced_options_elements + [
            sld_downsample,
            sld_hessian_sigma,
            self.lineedit_lambda_mse,
            self.lineedit_lambda_hessian,
            self.lineedit_lambda_tv,
            self.lineedit_n_epochs,
            sld_wedge,
            self.lineedit_inc,
            self.lineedit_neighbors,
        ]

        # QGroupBox
        parameters = QGroupBox("Parameters")
        gb_layout = QGridLayout()
        gb_layout.addWidget(label_vertical, 0, 0)
        gb_layout.addWidget(self.checkbox_vertical, 0, 1)
        gb_layout.addWidget(label_angle, 1, 0)
        gb_layout.addWidget(self.lineedit_angle, 1, 1, 1, 1)
        gb_layout.addWidget(self.angle_check_button, 1, 2, 1, -1)

        gb_layout.addWidget(self.label_guided_upsample_kernel_length, 2, 0)
        gb_layout.addWidget(sld_guided_upsample_kernel_length, 2, 1, 1, -1)
        gb_layout.addWidget(self.label_dark_stripe_only, 3, 0)
        gb_layout.addWidget(self.checkbox_dark_stripe_only, 3, 1, 1, -1)

        gb_layout.addWidget(label_mask, 4, 0)
        gb_layout.addWidget(self.combobox_mask, 4, 1, 1, -1)

        gb_layout.addWidget(label_backend, 5, 0)
        gb_layout.addWidget(self.combobox_baskend, 5, 1, 1, -1)
        gb_layout.addWidget(self.btn_advanced_options, 6, 0, 1, -1)
        gb_layout.addWidget(self.label_downsample_ratio, 7, 0)
        gb_layout.addWidget(sld_downsample, 7, 1, 1, -1)
        gb_layout.addWidget(self.label_hessian_kernel_sigma, 8, 0)
        gb_layout.addWidget(sld_hessian_sigma, 8, 1, 1, -1)
        gb_layout.addWidget(self.label_lambda, 9, 0, 3, 1)
        gb_layout.addWidget(self.label_lambda_hessian, 9, 1, 1, 1)
        gb_layout.addWidget(self.label_lambda_tv, 10, 1, 1, 1)
        gb_layout.addWidget(self.label_lambda_mse, 11, 1, 1, 1)
        gb_layout.addWidget(self.lineedit_lambda_hessian, 9, 2, 1, -1)
        gb_layout.addWidget(self.lineedit_lambda_tv, 10, 2, 1, -1)
        gb_layout.addWidget(self.lineedit_lambda_mse, 11, 2, 1, -1)
        gb_layout.addWidget(self.label_neighbors, 12, 0)
        gb_layout.addWidget(self.lineedit_neighbors, 12, 1, 1, -1)
        gb_layout.addWidget(self.label_inc, 13, 0)
        gb_layout.addWidget(self.lineedit_inc, 13, 1, 1, -1)
        gb_layout.addWidget(self.label_n_epochs, 14, 0)
        gb_layout.addWidget(self.lineedit_n_epochs, 14, 1, 1, -1)
        gb_layout.addWidget(self.label_wedge, 15, 0)
        gb_layout.addWidget(sld_wedge, 15, 1, 1, -1)

        gb_layout.setAlignment(Qt.AlignTop)
        parameters.setLayout(gb_layout)

        layout = QGridLayout()
        layout.addWidget(title, 0, 0, 1, -1)
        layout.addWidget(self.btn_load, 1, 0, 1, -1)
        layout.addWidget(parameters, 2, 0, 1, -1)
        layout.addWidget(btn_process, 3, 0)
        layout.addWidget(btn_save, 3, 1)

        widget = QWidget()
        widget.setFont(font)
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
            self.viewer.window._qt_window.closeEvent = lambda event: wrapper(
                self, func, event
            )

        self.viewer.layers.events.inserted.connect(self.update_combobox)
        self.viewer.layers.events.inserted.connect(self.connect_rename)
        self.viewer.layers.events.removed.connect(self.update_combobox)
        self.viewer.layers.events.reordered.connect(self.update_combobox)
        for layer in self.viewer.layers:
            layer.events.name.connect(self.update_combobox)
        self.update_combobox()

        self.logger.debug("DestripeWidget initialized")
        self.logger.info("Ready to use")

    def guided_upsample_kernel_length(self, value: int):
        self.lineedit_guided_upsample_kernel_length = value * 2 + 9
        self.label_guided_upsample_kernel_length.setText(
            "Upsample kernel: {}".format(
                self.lineedit_guided_upsample_kernel_length
            )
        )

    def backend_changed(self, index: int):
        if index == 0:
            self.backend = "jax"
        else:
            self.backend = "torch"

    def wedge(self, value: int):
        self.lineedit_wedge = value
        self.label_wedge.setText("Angular size of mask: {}".format(value))

    def downsample(self, value: int):
        self.lineedit_downsample = value
        self.label_downsample_ratio.setText(
            "Downsample ratio: {}".format(value)
        )

    def hessian_sigma(self, value: int):
        self.lineedit_hessian_sigma = float(value) * 0.5 + 0.5
        self.label_hessian_kernel_sigma.setText(
            "Hessian kernel sigma: %.1f" % (self.lineedit_hessian_sigma)
        )

    def lower_changed(self, value: int):
        # (19.11.2024)
        self.lower_percentage = float(value) / 100.0
        self.lbl_lower_percentage.setText(
            "lower percentage: %.2f" % (self.lower_percentage)
        )

    def toggle_input_display(self):
        layer_name = self.btn_load.currentText()
        print(layer_name)
        if layer_name != "":
            self.viewer.layers.move(self.viewer.layers.index(layer_name), -1)

    def update_combobox(self):
        self.logger.debug("Updating combobox...")
        layernames_label = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]
        layernames_image = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        layernames_label = layernames_label + ["None"]
        layernames_label.reverse()
        # current_text = self.btn_load.currentText()
        # if current_text in layernames_image:
        #     AllItems = [
        #         i
        #         for i in range(self.btn_load.count())
        #         if self.btn_load.itemText(i) != current_text
        #     ]
        #     print(AllItems)
        #     for item in AllItems:
        #         self.btn_load.removeItem(item)
        #     print(current_text)
        #     layernames_image.remove(current_text)
        #     print(layernames_image)
        #     self.btn_load.addItems(layernames_image)
        # else:
        #     self.btn_load.clear()
        #     self.btn_load.addItems(layernames_image)
        if set(layernames_image) != set(
            [self.btn_load.itemText(i) for i in range(self.btn_load.count())]
        ):
            self.btn_load.clear()
            self.btn_load.addItems(layernames_image)
        self.combobox_mask.clear()
        self.combobox_mask.addItems(layernames_label)

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
            if isinstance(layer, napari.layers.Image)
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
        model = DeStripe(
            resample_ratio=params["resample_ratio"],
            guided_upsample_kernel=params["guided_upsample_kernel"],
            hessian_kernel_sigma=params["hessian_kernel_sigma"],
            lambda_masking_mse=params["lambda_masking_mse"],
            lambda_tv=params["lambda_tv"],
            lambda_hessian=params["lambda_hessian"],
            inc=params["inc"],
            n_epochs=params["n_epochs"],
            wedge_degree=params["wedge_degree"],
            n_neighbors=params["n_neighbors"],
            backend=params["backend"],
        )
        output_image = model.train(
            is_vertical=params["is_vertical"],
            x=params["input_image"],
            mask=params["mask"],
            angle_offset=params["angle_offset"],
            display=False,
            non_positive=params["non_positive"],
            display_angle_orientation=False,
        )
        self.viewer.add_image(
            output_image,
            name="destriped image",
        )

    def get_parameters(self):
        params = {}
        mask_layer_name = self.combobox_mask.currentText()
        if mask_layer_name not in self.viewer.layers:
            if mask_layer_name != "None":
                self.logger.info("Selected mask not found")
                return
            else:
                params["mask"] = None
        else:
            mask_layer_index = self.viewer.layers.index(mask_layer_name)
            params["mask"] = self.viewer.layers[mask_layer_index].data

        input_layer_name = self.btn_load.currentText()
        if input_layer_name not in self.viewer.layers:
            self.logger.info("Selected input not found")
            return
        input_layer_index = self.viewer.layers.index(input_layer_name)
        params["input_image"] = self.viewer.layers[input_layer_index].data

        self.logger.debug(f"Selected mask: {mask_layer_name}")

        params["resample_ratio"] = self.lineedit_downsample
        params["guided_upsample_kernel"] = (
            self.lineedit_guided_upsample_kernel_length
        )
        params["hessian_kernel_sigma"] = float(self.lineedit_hessian_sigma)
        try:
            params["lambda_masking_mse"] = float(
                self.lineedit_lambda_mse.text()
            )
            params["lambda_tv"] = float(self.lineedit_lambda_tv.text())
            params["lambda_hessian"] = float(
                self.lineedit_lambda_hessian.text()
            )
        except Exception as e:
            self.logger.error(f"Invalid lambda parameters, error {e}")
            return
        try:
            params["inc"] = int(self.lineedit_inc.text())
        except Exception as e:
            self.logger.error(f"Invalid latent dimension, error {e}")
            return
        try:
            params["n_epochs"] = int(self.lineedit_n_epochs.text())
        except Exception as e:
            self.logger.error(f"Invalid training epochs, error {e}")
            return
        params["wedge_degree"] = self.lineedit_wedge
        try:
            params["n_neighbors"] = int(self.lineedit_neighbors.text())
        except Exception as e:
            self.logger.error(f"Invalid GNN neighbors, error {e}")
            return
        params["backend"] = self.backend

        params["is_vertical"] = self.checkbox_vertical.isChecked()

        self.logger.debug(f"Vertical: {params['is_vertical']}")
        try:
            params["angle_offset"] = list(
                map(float, self.lineedit_angle.text().split(","))
            )
        except ValueError:
            self.logger.error("Invalid angle offset")
            return
        self.logger.debug(f"Angle offset: {params['angle_offset']}")

        params["non_positive"] = self.checkbox_dark_stripe_only.isChecked()

        return params

    def toggle_check_angle(self):
        if self.angle_check_button.text() == "check":
            self.angle_check_button.setText("uncheck")

            is_vertical = self.checkbox_vertical.isChecked()
            angle_offset = list(
                map(float, self.lineedit_angle.text().split(","))
            )
            if self.viewer.layers[self.btn_load.currentText()].ndim == 3:
                z, m, n = self.viewer.layers[
                    self.btn_load.currentText()
                ].level_shapes[0]
            else:
                m, n = self.viewer.layers[
                    self.btn_load.currentText()
                ].level_shapes[0]
                z = 1
            if not is_vertical:
                (m, n) = (n, m)
            shapes_layer = self.viewer.add_shapes(name="angle offset(s)")
            for deg in angle_offset:
                d = np.tan(np.deg2rad(deg)) * m
                p0 = [0 + n // 2 - d // 2, d + n // 2 - d // 2]
                p1 = [0, m - 1]

                line_locations = []
                if is_vertical:
                    for s in range(z):
                        line_locations.append(
                            [[s, p1[1], p0[1]], [s, p1[0], p0[0]]]
                        )
                else:
                    for s in range(z):
                        line_locations.append(
                            [[s, p0[1], p1[1]], [s, p0[0], p1[0]]]
                        )
                shapes_layer.add(
                    np.asarray(line_locations),
                    shape_type="line",
                    face_color=[0] * 4,
                    edge_color="red",
                    edge_width=5,
                )

        else:
            self.angle_check_button.setText("check")
            try:
                self.viewer.layers.remove("angle offset(s)")
            except Exception as e:
                print(f"Error! {e}")
                pass

    def toggle_advanced_options(self):
        if self.btn_advanced_options.text() == "Show Advanced Parameters":
            self.btn_advanced_options.setText("Hide Advanced Parameters")
            show = True
        else:
            self.btn_advanced_options.setText("Show Advanced Parameters")
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
