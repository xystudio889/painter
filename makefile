app: app.py
	pip install pyinstaller>=6
	pyinstaller --noconfirm --clean --onedir --windowed --name "Number Scanner" --icon=icon.ico --add-data "mnist_cnn.pt;." --version-file version_info.txt app.py
