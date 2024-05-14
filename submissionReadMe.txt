### Instructions for using interacting with code found in the repository - Melric Lazaro (6695534) ###

Last Updated: 09/05/2024

### Environment Setup ###

Anaconda Python 3.9.18 is used for HumanAura on Windows 10 22H2 (NVIDIA Driver 527.99 - NVIDIA Geforce RTX 3060 6GB).

To use HumanAura, an environment needs to be setup containing the necessary libraries and dependencies.
It is suggested to use a Conda Environment to do so as this was used for setup on my device.

Two requirement files are included (requirements.txt and piprequirements.txt) for Conda and Pip environments. Please be aware that this might not work out of the box due to TensorFlow Object Detection API requiring a manual install and does not support pip.

It is highly suggested to follow the documentation found at https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html to create an environment from scratch as CUDA versions and CUDA Toolkit are specific to your device along with other dependencies.

Something important is that the right version of CUDNN needs to be installed for training.

ONNXRuntime GPU and ONNX Runtime Tools are also used for inferencing and modifying ONNX models.

### Additional Files ###

To run HumanAura inferencing out of the box without using the utility scripts, the additional files need to be downloaded from the additional GitHub Repository found at https://github.com/Melricflash/HumanAura-Additional/tree/main

Once downloaded, you can place these files into the original code archive and inferencing should work out of the box (please ensure paths are correct when running).

For running fine tuning processes with TensorFlow Object Detection API, the TFRecords and labelmap.pbtxt files are supplied. 

The pipeline files used for finetuning can be found in this code archive inside 'Fine Tuning Configs'.

For inferencing, 7 videos are supplied (6 from the subset dataset and 1 extra video). Additionally, the 4 ONNX models seen in the report are also supplied for inferencing.

The GitHub Repository used to store this code archive can be found at https://github.com/Melricflash/HumanAura
This is private at the time of writing but can become public if necessary, contact me if so.

### Utility Script Usage ###

### ExtractFrames.py ###

Supply the path to the video file to extract frames from in the variable.

The output will result in frames being deposited into a folder called 'frames', it is recommended to create the empty folder beforehand.

### XMLParser.py ###

Supply the path to the XML ground truth file found in the CAVIAR dataset, the output will create a text file containing the new annotations in the same folder with the specified file name.

### ConvertToTFRecord.py ###

Supply the path to the converted text file from XMLParser.py for corresponding video along with the path to the label_map.pbtxt file.
Supply the output name of the TFRecord to be generated.

Important note, these above files need to be located in the same directory where the video frames are stored for the script to work by default.

The output will be the TFRecord of the chosen name found in the same directory.

### visualiseTFRecord.py ###

Supply the path to the TFRecord you want to visualise.

The output will show a static frame from the TFRecord in MatPlotLib.

### HumanAura Testing ###

For each of these models, use the following videos to properly test the model with as these are unseen during training for the specific experiment:

MobileNet SSD V2 (mnet) - browse2.mpg, browse_while_waiting1.mpg, browse_while_waiting2.mpg
MobileNet SSD V2 + FPN (mnetFPN) - browse1.mpg, browse_while_waiting1.mpg, browse_while_waiting2.mpg

## Important - You must set isCnet = True in humanaura.py to inference with CentreNet ##
CentreNet + ResNet 50 (cnet) - browse4.mpg, browse_while_waiting2.mpg, browse2.mpg

EfficientDet D0 (efdet) - browse1.mpg, browse_while_waiting1.mpg, browse_while_waiting2.mpg

### Contact ###

If there are any issues with getting this program to work, please contact me at ml01663@surrey.ac.uk or melriclazaro@gmail.com :)