# HumanAura - Simple Human Detection & Tracking system across a CCTV Source (FYP)

HumanAura is a (FY) project designed to propose a solution for human detection and tracking **given limited resources** by using a 'lightweight' tracking algorithm. 

We utilise the TensorFlow Object Detection API and pre-trained models from the TensorFlow Object Detection Model Zoo to fine-tune the models using the CAVIAR dataset set in a retail scenario.

No models are supplied due to GitHub FS limits, but you are required to supply an **ONNX model** that has the **fixed input shape of [1,224,384,3]** to work (use ONNX utilities to convert the layer if needed)

Models currently tested with:
 - MobileNet SSD V2 (320x320)
 - MobileNet SSD V2 + FPN (320x320)
 - CentreNet ResNet 50
 - EfficientDet D0
