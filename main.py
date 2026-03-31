#!/usr/bin/env python3
"""
Microbe Analysis Dashboard - PyQt5 Main Window
Fixed button re-enable issue: Signal GUI update after script finish.
"""

import sys
import os
import subprocess
import threading
from queue import Queue, Empty
from datetime import datetime
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QEventLoop

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class OutputReader(QObject):
    line_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def add_output(self, text):
        self.queue.put(text)

    def process_queue(self):
        try:
            while True:
                text = self.queue.get_nowait()
                self.line_ready.emit(text)
        except Empty:
            pass

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(BASE_DIR, 'ui', 'main.ui'), self)
        
        self.output_reader = OutputReader()
        self.output_reader.line_ready.connect(self.append_console)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.output_reader.process_queue)
        self.timer.start(100)
        
        self.connect_buttons()
        self.is_running = False
        self.statusbar.showMessage('✅ Ready! Tabs 1→5 for workflow.')

    def connect_buttons(self):
        self.btn_generate_data.clicked.connect(lambda: self.run_script('generate_microbe_data_chatbot.py', '1. Generate Data'))
        self.btn_train_verbose.clicked.connect(lambda: self.run_script('microbe_training_verbose.py', '2. Train Model'))
        self.btn_demo_features.clicked.connect(lambda: self.run_script('ui/demo_features.py', '3. Feature Eng'))
        self.btn_analysis.clicked.connect(lambda: self.run_script('microbe_training_analysis.py', '4. Analysis'))
        self.btn_plots.clicked.connect(lambda: self.run_script('PCA-LDA-analysis_plots.py', '5. PCA/LDA Plots'))

    def append_console(self, text):
        cursor = self.textEdit_console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.textEdit_console.setTextCursor(cursor)
        self.textEdit_console.ensureCursorVisible()

    def disable_buttons(self):
        for btn_name in ['btn_generate_data', 'btn_train_verbose', 'btn_demo_features', 'btn_analysis', 'btn_plots']:
            getattr(self, btn_name).setEnabled(False)

    def enable_buttons(self):
        for btn_name in ['btn_generate_data', 'btn_train_verbose', 'btn_demo_features', 'btn_analysis', 'btn_plots']:
            getattr(self, btn_name).setEnabled(True)

    def run_script(self, script_name, title):
        if self.is_running:
            self.append_console('[INFO] Previous script running. Please wait.\n')
            return

        self.is_running = True
        self.statusbar.showMessage(f'Running {title}...')
        self.append_console(f'\n=== {title} START {datetime.now().strftime("%H:%M:%S")} ===\n')
        self.disable_buttons()

        script_path = os.path.join(BASE_DIR, script_name)

        def run_worker():
            try:
                process = subprocess.Popen(
                    ['python3', script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                for line in iter(process.stdout.readline, ''):
                    self.output_reader.add_output(line)

                rc = process.wait()
                self.output_reader.add_output(f'[DONE] Exit code: {rc}\n')
            except Exception as e:
                self.output_reader.add_output(f'[ERROR] {e}\n')
            finally:
                self.is_running = False
                self.output_reader.add_output('[READY] You can now click next button!\n')
                # Force GUI update
                QtWidgets.QApplication.processEvents()
                self.enable_buttons()

        threading.Thread(target=run_worker, daemon=True).start()

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

