# Raspberry Pi 4 Deployment

### Static Test Instructions (no camera needed)
- No OS requirement
- Make sure that all the images under `data/$quant` have been downsampled to 160x100 ($quant are the subdirectories of `quanteyes/data/`). If not, run `downsample.ipynb` with the correct file path in order to downsample the image sizes.
- Make sure that the desired `.tflite` file is present under `model_export/`, modify the path accordingly in tflm.py
- Finally, run the static test
```
python tflm.py
```

### Live Test Instructions (camera required)
- OS: [64-bit Raspberry Pi OS](https://www.raspberrypi.com/software/)
- Attach your raspberry pi camera, and boot your Pi
- Follow the instruction to install [pycamera2](https://github.com/raspberrypi/picamera2) (pip recommended)
- Install other Python requirements by running
```
pip install -r requirements.txt
```
- If you run into an error with pip begin externally managed, run the following command ([source](https://stackoverflow.com/questions/75608323/how-do-i-solve-error-externally-managed-environment-every-time-i-use-pip-3)):
```
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED
```
- Finally run the live test:
```
python camera.py
```
