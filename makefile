app: app.py
	pip install pyinstaller>=6
	pyinstaller --noconfirm --clean --onedir --windowed --name "Number Scanner" --icon=icon.ico --version-file version_info.txt app.py
	xcopy /E /I /Y models "dist\Number Scanner\models"
