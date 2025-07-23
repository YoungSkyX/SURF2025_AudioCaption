from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFileDialog, QStackedWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon, QPixmap
import sys
import os
import wave
import threading
import time
import datetime

# Import inference
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference'))   #
try:
    from inference import inference
except Exception as e:
    inference = None
    print(f"inference import error: {e}")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

RECORD_DIR = os.path.join(os.path.dirname(__file__), 'recoding')
if not os.path.exists(RECORD_DIR):
    os.makedirs(RECORD_DIR)

# Rules for the WAV file naming

def get_next_wav_filename():
    files = [f for f in os.listdir(RECORD_DIR) if f.endswith('.wav')]
    nums = [int(f.split('.')[0]) for f in files if f.split('.')[0].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(RECORD_DIR, f"{next_num:03d}.wav")

class DragDropWidget(QFrame):
    ai_caption = pyqtSignal(str)
    file_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.label = QLabel("Drag or select a .wav file here", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFont(QFont("Tahoma", 14))
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.wav'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.wav'):
                self.save_wav(file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("WAV Files (*.wav)")
            if file_dialog.exec():
                file_path = file_dialog.selectedFiles()[0]
                self.save_wav(file_path)

    def save_wav(self, src_path):
        dst_path = get_next_wav_filename()
        with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
            dst.write(src.read())
        self.file_saved.emit(f"File saved to: {dst_path}")
        if inference:
            def run_infer():
                self.file_saved.emit("Analyzing audio, please wait...")
                try:
                    result = inference(dst_path)
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.ai_caption.emit(f"{now} AI Caption: {result}")
                except Exception as e:
                    self.ai_caption.emit(f"AI inference error: {e}")
            threading.Thread(target=run_infer, daemon=True).start()

class RecordWidget(QFrame):
    record_status = pyqtSignal(str)
    record_time = pyqtSignal(str)
    caption_result = pyqtSignal(str)  # Signal for AI caption result

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout = QVBoxLayout(self)
        self.record_btn = QPushButton("Record", self)
        self.record_btn.setFont(QFont("Tahoma", 13))
        self.record_btn.clicked.connect(self.start_record)
        self.layout.addWidget(self.record_btn)
        self.setLayout(self.layout)
        self.is_recording = False
        self.is_paused = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.elapsed_seconds = 0
        self.time_label = QLabel("00:00", self)
        self.time_label.setFont(QFont("Consolas", 13))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.time_label)
        self.time_label.hide()
        # Recording variables
        self.audio_thread = None
        self.audio_stream = None
        self.audio_frames = []
        self.wav_path = None
        self.p = None
        self.lock = threading.Lock()

    def start_record(self):
        if not PYAUDIO_AVAILABLE:
            self.record_status.emit("pyaudio is not installed, cannot record!")
            return
        self.record_btn.hide()
        self.pause_btn = QPushButton("Pause", self)
        self.pause_btn.setFont(QFont("Tahoma", 12))
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setFont(QFont("Tahoma", 12))
        self.stop_btn.setStyleSheet("background-color: #FF6B6B; color: #FFFFFF;")
        self.pause_btn.clicked.connect(self.pause_record)
        self.stop_btn.clicked.connect(self.stop_record)
        self.layout.addWidget(self.pause_btn)
        self.layout.addWidget(self.stop_btn)
        self.is_recording = True
        self.is_paused = False
        self.elapsed_seconds = 0
        self.time_label.setText("00:00")
        self.time_label.show()
        self.timer.start(1000)
        self.wav_path = get_next_wav_filename()
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()
        self.record_status.emit(f"Start recording... File will be saved to: {self.wav_path}")

    def update_time(self):
        if self.is_recording and not self.is_paused:
            self.elapsed_seconds += 1
            m, s = divmod(self.elapsed_seconds, 60)
            self.time_label.setText(f"{m:02d}:{s:02d}")
            self.record_time.emit(self.time_label.text())

    def record_audio(self):
        self.p = pyaudio.PyAudio()
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=32000, input=True, frames_per_buffer=1024)
        self.audio_stream = stream
        while self.is_recording:
            if self.is_paused:
                time.sleep(0.1)
                continue
            data = stream.read(1024, exception_on_overflow=False)
            with self.lock:
                self.audio_frames.append(data)
        stream.stop_stream()
        stream.close()
        self.p.terminate()
        # Save wav
        wf = wave.open(self.wav_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(32000)
        with self.lock:
            for frame in self.audio_frames:
                wf.writeframes(frame)
        wf.close()
        self.record_status.emit(f"Recording saved: {os.path.basename(self.wav_path)}")
        # After recording, automatically run inference
        if inference:
            # Use thread to avoid blocking UI
            def run_infer():
                self.record_status.emit("Analyzing audio, please wait...")
                try:
                    result = inference(self.wav_path)
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.caption_result.emit(f"{now} AI Caption: {result}")
                except Exception as e:
                    self.caption_result.emit(f"AI inference error: {e}")
            threading.Thread(target=run_infer, daemon=True).start()

    def pause_record(self):
        if not self.is_paused:
            self.pause_btn.setText("Resume")
            self.is_paused = True
            self.record_status.emit("Recording paused")
        else:
            self.pause_btn.setText("Pause")
            self.is_paused = False
            self.record_status.emit("Recording resumed...")

    def stop_record(self):
        self.pause_btn.deleteLater()
        self.stop_btn.deleteLater()
        self.record_btn.show()
        self.is_recording = False
        self.is_paused = False
        self.timer.stop()
        self.time_label.hide()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join()
        self.record_status.emit("Recording stopped")

class StartPage(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        self.left_output = None
        
        # Output area with image overlay
        output_container = QFrame(self)
        output_container.setLayout(QVBoxLayout())
        output_container.layout().setContentsMargins(0, 0, 0, 0)
        
        # Image label
        self.image_label = QLabel(self)
        try:
            pixmap = QPixmap(os.path.join(os.path.dirname(__file__), "IMG.png"))
            if not pixmap.isNull():
                # Scale image to fit the container while maintaining aspect ratio - made larger
                scaled_pixmap = pixmap.scaled(600, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Image not found")
        except Exception as e:
            self.image_label.setText("Failed to load image")
        
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 6px; }")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Text output box
        self.output_box = QTextEdit(self)
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("AI output...")
        self.output_box.setFont(QFont("Tahoma", 13))
        self.output_box.setStyleSheet("QTextEdit { color: #87CEFA; }")
        self.output_box.hide()  # Initially hidden
        
        # Add both to container
        output_container.layout().addWidget(self.image_label)
        output_container.layout().addWidget(self.output_box)
        
        # Bottom area
        bottom_layout = QHBoxLayout()
        self.record_widget = RecordWidget(self)
        self.dragdrop_widget = DragDropWidget(self)
        bottom_layout.addWidget(self.record_widget, 1)
        bottom_layout.addWidget(self.dragdrop_widget, 3)
        main_layout.addWidget(output_container, 2)
        main_layout.addLayout(bottom_layout, 1)
        self.setLayout(main_layout)
        # Connect signals
        self.record_widget.record_status.connect(self.append_left_output)
        self.record_widget.record_time.connect(self.show_time)
        self.record_widget.caption_result.connect(self.append_ai_output)
        self.dragdrop_widget.ai_caption.connect(self.append_ai_output)
        self.dragdrop_widget.file_saved.connect(self.append_left_output)

    def set_left_output(self, left_output_widget):
        self.left_output = left_output_widget

    def append_left_output(self, msg):
        if self.left_output:
            html = f'<span style="color:#234567;">{msg}</span>'
            self.left_output.append(html)

    def append_ai_output(self, msg):
        # When there's output, hide image and show text
        self.image_label.hide()
        self.output_box.show()
        html = f'<span style="color:#1976D2;">{msg}</span>'
        self.output_box.append(html)

    def show_time(self, t):
        pass

class AboutPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("About AudioCaption", self)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Tahoma", 15, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            " AudioCaption is an intelligent audio captioning tool.\n\n"
            "You can record or drag-and-drop a .wav file, and the software will automatically generate a caption for the audio using an AI model.\n\n"
            "How it works:\n"
            "  1. The software records or receives a .wav audio file.\n"
            "  2. The audio is processed and sent to a deep learning model (LLaMA or Qwen) for inference.\n"
            "  3. The model analyzes the audio content and generates a descriptive caption.\n"
            "  4. The result is displayed in the output area.\n\n"
            "This tool is designed for research, accessibility, and creative applications, making audio content more accessible and understandable.\n\n"
            "--Developed with PyQt6 and Pytorch."
        , self)
        desc.setWordWrap(True)
        desc.setFont(QFont("Tahoma", 11))
        layout.addWidget(desc)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioCaption")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "icon.png")))
        self.resize(800, 500)
        main_widget = QWidget(self)
        main_layout = QHBoxLayout(main_widget)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #F5F7FA;
            }
            QFrame#leftFrame {
                background: #4A90E2;
                border: none;
            }
            QLabel#titleLabel {
                color: #FFFFFF;
            }
            QPushButton {
                background: #6BB6FF;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background: #357ABD;
            }
            QStackedWidget, QWidget#rightArea {
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
            QTextEdit {
                background: #FFFFFF;
                color: #333333;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
            }
            QLabel, QFrame, QVBoxLayout, QHBoxLayout {
                color: #333333;
            }
        """)
##
        # Left area
        left_layout = QVBoxLayout()
        # Title
        title_label = QLabel("Audio\nCaption", self)
        title_label.setFont(QFont("Tahoma", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title_label)
        # Button area 1
        btn1_frame = QFrame(self)
        btn1_layout = QVBoxLayout(btn1_frame)
        self.start_btn = QPushButton("Start", self)
        self.start_btn.setFont(QFont("Tahoma", 14))
        btn1_layout.addWidget(self.start_btn)
        btn1_frame.setLayout(btn1_layout)
        left_layout.addWidget(btn1_frame)
        # Divider
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(line)
        # Button area 2
        btn2_frame = QFrame(self)
        btn2_layout = QVBoxLayout(btn2_frame)
        self.About_btn = QPushButton("About", self)
        self.About_btn.setFont(QFont("Tahoma", 14))
        btn2_layout.addWidget(self.About_btn)
        btn2_frame.setLayout(btn2_layout)
        left_layout.addWidget(btn2_frame)
        left_layout.addStretch(1)
        # Left output
        left_layout.addStretch(1)
        
        # Add toggle button and output layout
        output_control_layout = QHBoxLayout()
        # Add round toggle button
        self.toggle_output_btn = QPushButton("", self)
        self.toggle_output_btn.setFixedSize(20, 20)
        self.toggle_output_btn.setStyleSheet("""
            QPushButton {
                background-color: #87CEFA;
                border-radius: 10px;
                border: none;
            }
            QPushButton:checked {
                background-color: transparent;
                border: 2px solid #CCCCCC;
                border-radius: 10px;
            }
        """)
        self.toggle_output_btn.setCheckable(True)
        self.toggle_output_btn.clicked.connect(self.toggle_output)
        self.toggle_output_btn.setToolTip("Hide system output")
        # Add button to the left
        output_control_layout.addWidget(self.toggle_output_btn)
        output_control_layout.addStretch(1)
        left_layout.addLayout(output_control_layout)
        
        self.left_output = QTextEdit(self)
        self.left_output.setReadOnly(True)
        self.left_output.setStyleSheet("background:#e6f0fa; color:#234567; border-radius:5px; font-size:12px;")
        self.left_output.setPlaceholderText("System output...")
        self.left_output.setFont(QFont("Tahoma", 13))
        left_layout.addWidget(self.left_output)
        left_frame = QFrame(self)
        left_frame.setLayout(left_layout)
        left_frame.setFixedWidth(160)

        # Right stacked area
        self.stack = QStackedWidget(self)
        self.start_page = StartPage()
        self.About_page = AboutPage()
        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.About_page)
        main_layout.addWidget(left_frame, 1)
        main_layout.addWidget(self.stack, 2)
        self.setCentralWidget(main_widget)
        self.start_page.set_left_output(self.left_output)
        self.start_btn.clicked.connect(self.show_start)
        self.About_btn.clicked.connect(self.show_about)
        self.show_start()

    def show_start(self):
        self.stack.setCurrentWidget(self.start_page)

    def show_about(self):
        self.stack.setCurrentWidget(self.About_page)
        
    def toggle_output(self):
        if self.toggle_output_btn.isChecked():
            self.left_output.hide()
            self.toggle_output_btn.setToolTip("Show system output")
        else:
            self.left_output.show()
            self.toggle_output_btn.setToolTip("Hide system output")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
