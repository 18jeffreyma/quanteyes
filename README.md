# quanteyes
Quantization of SOTA models for gaze tracking and exploring development of custom processing units to perform quantization in HW.

# Development Setup
1. Create and import a conda environconda with required dependencies: `conda env create -f env.yml --name <YOUR_ENV_NAME>`
2. Activate new conda environment: `conda activate <YOUR_ENV_NAME>`
3. Install this repo package as a local module symbolically (assuming cwd is this repository): `pip install -e .`

# Model Development
1. run `python quanteyes/training_tf/train_quantized.py` to train the models using QAT and quantize/export the models.
2. run `python quanteyes/training_tf/evaluate_quantized.py` to evaluate the quantized models using TFLite interpreter.
