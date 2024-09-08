import sys
import os
import toml
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QLabel, QTextEdit, QPushButton, QFileDialog,
                             QSplitter, QScrollArea, QDialog)
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtCore import Qt, QSize
import paramiko
import os
import stat

import paramiko
import os
import stat
import time


def download_directory_sftp(hostname, port, username, password, remote_directory, local_directory):
    # 创建SSH客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动添加密钥

    try:
        # 连接服务器
        ssh.connect(hostname, port=port, username=username, password=password)
        sftp = ssh.open_sftp()

        # 创建本地目录
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        # 定义一个递归函数来下载整个目录
        def download_recursive(remote_path, local_path):
            try:
                # 获取远程目录中的所有文件和子目录
                files = sftp.listdir_attr(remote_path)

                for file in files:
                    remote_file_path = remote_path + '/' + file.filename  # 手动拼接路径，使用正斜杠
                    local_file_path = os.path.join(local_path, file.filename)  # 本地路径使用系统的os.path.join处理

                    # 如果是文件夹，递归下载
                    if stat.S_ISDIR(file.st_mode):
                        # 需求1：如果本地存在同名子文件夹，跳过
                        if os.path.exists(local_file_path):
                            print(f"Skipping directory {local_file_path} because it already exists.")
                        else:
                            os.makedirs(local_file_path)
                            print(f"Downloading directory: {remote_file_path}")
                            download_recursive(remote_file_path, local_file_path)

                        # 需求2：设置本地文件夹的修改时间与服务器一致
                        remote_mtime = file.st_mtime
                        os.utime(local_file_path, (remote_mtime, remote_mtime))

                    else:
                        # 下载文件
                        print(f"Downloading file: {remote_file_path} to {local_file_path}")
                        sftp.get(remote_file_path, local_file_path)

                        # 需求2：设置本地文件的修改时间与服务器一致
                        remote_mtime = file.st_mtime
                        os.utime(local_file_path, (remote_mtime, remote_mtime))

            except Exception as e:
                print(f"Failed to download directory {remote_path}: {e}")

        # 开始递归下载
        download_recursive(remote_directory, local_directory)

        print("Download completed.")

    except Exception as e:
        print(f"Failed to connect or download: {e}")

    finally:
        # 关闭SFTP会话和SSH连接
        if 'sftp' in locals():
            sftp.close()
        ssh.close()



class ScalableImageLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_pixmap = None

    def setScaledPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.updateScaledPixmap()

    def updateScaledPixmap(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateScaledPixmap()

class ExperimentViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实验结果查看器")
        self.setGeometry(100, 100, 1400, 900)

        # 主布局
        main_layout = QHBoxLayout()

        # 左侧实验列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.experiment_list = QListWidget()
        self.experiment_list.itemClicked.connect(self.load_experiment)
        left_layout.addWidget(QLabel("实验列表:"))
        left_layout.addWidget(self.experiment_list)

        # 右侧信息显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 图像显示区域
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_widget = QWidget()
        self.image_layout = QHBoxLayout(self.image_widget)
        self.original_image = ScalableImageLabel()
        self.reconstructed_image = ScalableImageLabel()
        self.comparison_image = ScalableImageLabel()
        self.image_layout.addWidget(self.original_image)
        self.image_layout.addWidget(self.reconstructed_image)
        # self.image_layout.addWidget(self.comparison_image)
        self.image_scroll_area.setWidget(self.image_widget)

        # 下方信息区域
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)

        # 实验摘要
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(QLabel("实验摘要:"))
        summary_layout.addWidget(self.summary_text)

        # 评估结果
        evaluation_widget = QWidget()
        evaluation_layout = QVBoxLayout(evaluation_widget)
        self.evaluation_text = QTextEdit()
        self.evaluation_text.setReadOnly(True)
        evaluation_layout.addWidget(QLabel("评估结果:"))
        evaluation_layout.addWidget(self.evaluation_text)

        info_layout.addWidget(summary_widget)
        info_layout.addWidget(evaluation_widget)

        # 查看配置按钮
        self.view_config_button = QPushButton("查看配置文件")
        self.view_config_button.clicked.connect(self.view_config)

        # 将所有组件添加到右侧布局
        right_layout.addWidget(self.image_scroll_area, 7)  # 图像区域占70%
        right_layout.addWidget(info_widget, 2)  # 信息区域占20%
        right_layout.addWidget(self.view_config_button, 1)  # 按钮占10%

        # 使用QSplitter来允许用户调整左右两侧的宽度
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 加载实验列表
        self.load_experiment_list()

    def load_experiment_list(self):
        # base_dir = QFileDialog.getExistingDirectory(self, "选择实验结果根目录")
        base_dir = "C:/Users/syb/OneDrive - SYB/实验/res"
        if base_dir:
            for entry in os.scandir(base_dir):
                if entry.is_dir() and entry.name.startswith("experiment_"):
                    self.experiment_list.addItem(entry.name)

    def load_experiment(self, item):
        experiment_dir = os.path.join(os.path.dirname(self.experiment_list.item(0).text()), item.text())
        # 首先判断需要的文件是否都存在,只要有一种文件不存在,就删除当前的 experiment_dir,并刷新界面的列表
        if not os.path.exists(os.path.join(experiment_dir, 'experiment_summary.txt')):
            # 即使目录非空,也强制删除 experiment_dir
            os.system(f"rd /s /q {experiment_dir}")
            self.experiment_list.clear()
            self.load_experiment_list()
            return


        # 加载摘要
        with open(os.path.join(experiment_dir, 'experiment_summary.txt'), 'r') as f:
            self.summary_text.setText(f.read())

        # 加载图像
        self.load_image(os.path.join(experiment_dir, 'original_image.png'), self.original_image, "原始图像")
        self.load_image(os.path.join(experiment_dir, 'reconstructed_image.png'), self.reconstructed_image, "重建图像")
        # self.load_image(os.path.join(experiment_dir, 'comparison.png'), self.comparison_image, "比较图像")

        # 加载评估结果
        with open(os.path.join(experiment_dir, 'evaluation_results.toml'), 'r') as f:
            evaluation_data = toml.load(f)
            self.evaluation_text.setText(str(evaluation_data))

        self.current_experiment_dir = experiment_dir

    def load_image(self, image_path, label, title):
        pixmap = QPixmap(image_path)
        label.setScaledPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setToolTip(title)

    def view_config(self):
        if hasattr(self, 'current_experiment_dir'):
            config_path = os.path.join(self.current_experiment_dir, 'config.toml')
            with open(config_path, 'r') as f:
                config_data = toml.load(f)
                config_text = toml.dumps(config_data)
                self.show_config_dialog(config_text)

    def show_config_dialog(self, config_text):
        dialog = QDialog(self)
        dialog.setWindowTitle("配置文件")
        dialog.setGeometry(200, 200, 400, 300)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(config_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        dialog.exec()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        # 在窗口调整大小时更新图像
        self.original_image.updateScaledPixmap()
        self.reconstructed_image.updateScaledPixmap()
        self.comparison_image.updateScaledPixmap()

if __name__ == "__main__":
    # 使用函数
    hostname = "10.140.33.51"
    port = 22  # SFTP的默认端口
    username = "liu_iot2024_b"
    password = (".Ss13626350673")  # 确保密码正确
    remote_directory = "/home/work/workspace/liu_iot2024_b/projects/MyINR/res"  # 使用正斜杠
    local_directory = r"C:\Users\syb\OneDrive - SYB\实验\res"

    download_directory_sftp(hostname, port, username, password, remote_directory, local_directory)
    app = QApplication(sys.argv)
    viewer = ExperimentViewer()
    viewer.show()
    sys.exit(app.exec())(myinr)