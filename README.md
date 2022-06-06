# find-object
run:
```
pyuic5 windows.ui -o gui.py
python3 find-object.py
```
### choose objects:
edit main.py, line 29:
``` image_paths = ["images/object_01.png", "images/object_02.png", "images/object_demo.jpg"]```
### choose camera-id:
edit VideoThread, run() camera_index
