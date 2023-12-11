# Raspberry Pi Zero W Deployment

### Static Test Instructions (no camera needed)
- No OS requirement
- Make sure that all the images under `data/$quant` have been downsampled to 160x100 ($quant are the subdirectories of `quanteyes/data/`). If not, run `downsample.ipynb` with the correct file path in order to downsample the image sizes.
- Make sure that the desired `.tflite` file is present under `model_export/`, modify the path accordingly in tflm.py
- Finally, run the static test
```
python tflm.py
```
