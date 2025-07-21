"""
3D RIR Analysis

This software analyzes 4 mono audio files provided by an Ambisonics B-Format recording
and returns:

- Omnidirectional (W) analysis (EDT, RT20, RT30, C50, C80, D50, SNR, and center time)
by octave or 1/3 octave band.
- Export a csv with intensity peaks (time and angle of incidence) for spatial analysis.
- Floor plan (provided as .jpg)  with an overlay of a hedgehog plot of incoming directions.
- Interactive matplotlib object for exploration of the 3D plot (not exported).

The output includes:
- W Analysis: `Acoustics.csv`, 'Acoustics.xlsx', 'Acoustics.png' and `W_Smoothed_Curves.png`.
- A format: BLD_IR.wav, FLU_IR.wav, FRD_IR.wav, BRU_IR.wav.
- B format: W_IR.wav, X_IR.wav, Y_IR.wav, Z_IR.wav
- `intensity_table.csv` with detected peak data.
- `floorplan_with_overlay.png`. Floorplan with hedgehog overlay.
- A '.png' with front, top, side and isometric view of the hedgehog.

Usage:
    python mainGUI.py

Requirements:
    matplotlib==3.10.3
    mplcursors==0.6
    numpy==2.3.1
    openpyxl==3.1.5
    pandas==2.3.1
    Pillow==11.3.0
    plotly==6.2.0
    PyQt5==5.15.11
    PyQt5_sip==12.17.0
    scipy==1.16.0
    soundfile==0.13.1

Author:
    [López Di Benedetto, Miguel Ángel]
    [2025]
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QTabWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QFormLayout,
    QMessageBox, QComboBox, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from mainProcessing import main_processing
import pandas as pd
import os
import subprocess

# Main GUI, pretty much self-explainatory.
class MainGUI(QWidget):
    # UI.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select 5 audio files and floor plan")
        self.setFixedSize(750, 750)

        self.audio_paths = [None] * 5
        self.image_path = None
        self.integration_time_ms = 20
        self.threshold_db = -40

        self.font_button = QFont('Arial', 14)
        self.font_label = QFont('Arial', 14)

        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-family: Arial;
                font-size: 14pt;
                font-weight: normal;
                padding: 5px 10px;
                min-width: 250px;
                min-height: 40px;
            }
        """)
        self.tabs.setGeometry(0, 0, 800, 720)

        self.tab_audio = QWidget()
        self.tab_floorplan = QWidget()

        self.tabs.addTab(self.tab_audio, "Audios (B-Format)")
        self.tabs.addTab(self.tab_floorplan, "Floor Plan Data")

        self.create_audio_tab()
        self.create_floorplan_tab()

    def create_audio_tab(self):
        layout = QVBoxLayout(self.tab_audio)
        self.buttons = []
        self.labels = []

        capsule_info = [
            ("FLU", "Red"),
            ("FRD", "Yellow"),
            ("BLD", "Green"),
            ("BRU", "Blue")
        ]

        for i in range(4):
            row = QHBoxLayout()
            btn = QPushButton(f"Load Audio {i+1}")
            btn.setFont(self.font_button)
            btn.setFixedSize(220, 50)
            btn.clicked.connect(lambda _, idx=i: self.load_audio(idx))

            channel, color = capsule_info[i]
            label = QLabel(f"{channel} - {color}: No audio")
            label.setFont(self.font_label)
            label.setFixedSize(500, 50)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            self.buttons.append(btn)
            self.labels.append(label)

            row.addWidget(btn)
            row.addWidget(label)
            layout.addLayout(row)

        row_inv = QHBoxLayout()
        btn_inv = QPushButton("Inverse Filter")
        btn_inv.setFont(self.font_button)
        btn_inv.setFixedSize(220, 50)
        btn_inv.clicked.connect(lambda _, idx=4: self.load_audio(idx))

        label_inv = QLabel("Inverse Filter: no audio")
        label_inv.setFont(self.font_label)
        label_inv.setFixedSize(500, 50)
        label_inv.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.buttons.append(btn_inv)
        self.labels.append(label_inv)

        row_inv.addWidget(btn_inv)
        row_inv.addWidget(label_inv)
        layout.addLayout(row_inv)
        # Band Filter selection
        row_band_filter = QHBoxLayout()
        band_filter_label = QLabel("Band Filter:")
        band_filter_label.setFont(self.font_label)
        row_band_filter.addWidget(band_filter_label)
        
        self.band_filter_combo = QComboBox()
        self.band_filter_combo.addItems(["Octave Band", "1/3 Octave Band"])
        self.band_filter_combo.setCurrentText("Octave Band")
        self.band_filter_combo.setFixedWidth(200)
        self.band_filter_combo.setFont(self.font_label)
        self.band_filter = self.band_filter_combo.currentText()
        self.band_filter_combo.currentTextChanged.connect(lambda val: setattr(self, 'band_filter', val))
        row_band_filter.addWidget(self.band_filter_combo)
        row_band_filter.addStretch()
        layout.addLayout(row_band_filter)
        layout.addStretch()
     
    def create_floorplan_tab(self):
        layout = QVBoxLayout(self.tab_floorplan)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
    
        # ---- Floor Plan Image ----
        row_image = QHBoxLayout()
        row_image.addStretch()
    
        self.btn_load_image = QPushButton("Floor Plan (jpg)")
        self.btn_load_image.setFont(self.font_button)
        self.btn_load_image.setFixedSize(220, 50)
        self.btn_load_image.clicked.connect(self.load_image)
    
        self.label_image = QLabel("No image")
        self.label_image.setFont(self.font_label)
        self.label_image.setFixedSize(500, 50)
        self.label_image.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    
        row_image.addWidget(self.btn_load_image)
        row_image.addWidget(self.label_image)
        row_image.addStretch()
        layout.addLayout(row_image)
    
        self.image_preview = QLabel()
        self.image_preview.setFixedSize(320, 240)
        self.image_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_preview, alignment=Qt.AlignCenter)
    
        # ---- Grid Layout for Dimensions and Mic Position ----
        container = QWidget()
        container.setMaximumWidth(500)
        container.setMinimumWidth(480)
        grid = QGridLayout(container)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)
    
        # Dimensions Title
        label_dims_title = QLabel("Dimensions:")
        label_dims_title.setFont(self.font_label)
        grid.addWidget(label_dims_title, 0, 0)
    
        # Width
        width_label = QLabel("Width:")
        width_label.setFont(self.font_label)
        self.input_dim_x = QLineEdit()
        self.input_dim_x.setFixedWidth(60)
        grid.addWidget(width_label, 0, 1)
        grid.addWidget(self.input_dim_x, 0, 2)
    
        # Height
        length_label = QLabel("Length:")
        length_label.setFont(self.font_label)
        self.input_dim_y = QLineEdit()
        self.input_dim_y.setFixedWidth(60)
        grid.addWidget(length_label, 0, 3)
        grid.addWidget(self.input_dim_y, 0, 4)
    
        # Mic Position Title
        label_pos_title = QLabel("Mic Position:")
        label_pos_title.setFont(self.font_label)
        grid.addWidget(label_pos_title, 1, 0)
    
        # X Position
        x_label = QLabel("X:")
        x_label.setFont(self.font_label)
        self.input_pos_x = QLineEdit()
        self.input_pos_x.setFixedWidth(60)
        grid.addWidget(x_label, 1, 1)
        grid.addWidget(self.input_pos_x, 1, 2)
    
        # Y Position
        y_label = QLabel("Y:")
        y_label.setFont(self.font_label)
        self.input_pos_y = QLineEdit()
        self.input_pos_y.setFixedWidth(60)
        grid.addWidget(y_label, 1, 3)
        grid.addWidget(self.input_pos_y, 1, 4)
    
        layout.addWidget(container)
    
        # Integration Time
        row_integration = QHBoxLayout()
        integration_label = QLabel("Integration Time (ms):")
        integration_label.setFont(self.font_label)
        row_integration.addWidget(integration_label)
        self.integration_combo = QComboBox()
        self.integration_combo.addItems(["1", "5", "10", "15", "20"])
        self.integration_combo.setCurrentText("10")
        self.integration_combo.setFixedWidth(100)
        self.integration_combo.currentTextChanged.connect(lambda val: setattr(self, 'integration_time_ms', int(val)))
        row_integration.addWidget(self.integration_combo)
        row_integration.addStretch()
        layout.addLayout(row_integration)
    
        # Threshold
        row_threshold = QHBoxLayout()
        threshold_label = QLabel("Threshold (dB):")
        threshold_label.setFont(self.font_label)
        row_threshold.addWidget(threshold_label)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["-10", "-20", "-30", "-40", "-50", "-60"])
        self.threshold_combo.setCurrentText("-20")
        self.threshold_combo.setFixedWidth(100)
        self.threshold_combo.currentTextChanged.connect(lambda val: setattr(self, 'threshold_db', int(val)))
        row_threshold.addWidget(self.threshold_combo)
        row_threshold.addStretch()
        layout.addLayout(row_threshold)
    
        # ---- Process Button ----
        self.btn_process = QPushButton("Process")
        self.btn_process.setFont(self.font_button)
        self.btn_process.setFixedSize(220, 50)
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)
    
        layout.addStretch()

    def load_audio(self, idx):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Audio File {idx+1}" if idx < 4 else "Select Inverse Filter Audio File",
            "",
            "Audio files (*.wav *.aiff *.flac)"
        )
        if path:
            self.audio_paths[idx] = path
            self.labels[idx].setText(path.split("/")[-1])

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Floor Plan Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.image_path = path
            self.label_image.setText(path.split("/")[-1])
            pixmap = QPixmap(path)
            self.image_preview.setPixmap(pixmap.scaled(
                self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
    
    def start_processing(self):
        if (None in self.audio_paths or
            not self.image_path or
            not self.input_dim_x.text().strip() or
            not self.input_dim_y.text().strip() or
            not self.input_pos_x.text().strip() or
            not self.input_pos_y.text().strip()):
    
            QMessageBox.warning(self, "Error", "Missing data")
            return
    
        # Output folder
        original_dir = os.getcwd()
        output_dir = os.path.join(original_dir, "Output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate user input dataframe
        userInput = pd.DataFrame([{
            "FLU": self.audio_paths[0],
            "FRD": self.audio_paths[1],
            "BLD": self.audio_paths[2],
            "BRU": self.audio_paths[3],
            "InverseFilter": self.audio_paths[4],
            "FloorPlan": self.image_path,
            "Dim_X": self.input_dim_x.text(),
            "Dim_Y": self.input_dim_y.text(),
            "Pos_X": self.input_pos_x.text(),
            "Pos_Y": self.input_pos_y.text(),
            "IntegrationTime_ms": self.integration_time_ms,
            "Threshold_dB": self.threshold_db,
            "Band_Filter": self.band_filter
        }])
    
        try:
            os.chdir(output_dir)
            main_processing(userInput)
        finally:
            os.chdir(original_dir)
    
        # Show Output folder
        subprocess.Popen(f'explorer "{output_dir}"')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainGUI()
    window.show()
    sys.exit(app.exec_())
