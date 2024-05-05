+++
title = 'Real-time Object Recognition in Chess: Personalized Tuning and Hardware Acceleration'
date = 2023-08-05T16:53:11-05:00
draft = false
categories = ['Computer Vision']
tags = ['Deep Learning', 'Project', 'yolov5']
+++

#### 1. Selected and customized the YOLOv5 model for Chinese chess annotation data.

<img src="https://s2.loli.net/2024/01/27/zASf74aWbsmn5tP.png">

#### 2. Conducted testing and analysis of the model.

<img src="https://s2.loli.net/2024/01/27/7f4cP9gOke5Zy3W.png">

The results indicated exceptional accuracy in recognition capabilities. However, a significant shortfall was identified in terms of efficiency, with the model taking approximately 6 seconds to process a single image.

#### 3. Implemented model optimization.

We substitute the YOLOv5 model with a more lightweight variant, YOLOv5-lite and convert the model into the ONNX format to leverage hardware acceleration, thereby enhancing computational efficiency.

#### 4. Deployed the model on edge computing devices using ONNXRUNTIME.

<img src="https://s2.loli.net/2024/01/27/wtzKNoyCkMFufVI.png">

This deployment resulted in maintaining a similar rate of recognition success while significantly increasing the frame rate of recognition to 6-8 frames per second, an efficiency improvement of approximately 97%. (The rate of rocognotion of b_zu is due to these set of chess is not made of wood with word engraved in it, meaning it is better change a set of chess whose texture is not affected by reflection)

#### 5. Future plans

Future plans are aimed at further enhancing system performance and expanding functionalities. This includes:

- Integrating a TPU acceleration stick (with subsequent conversion to OpenVino) or switching to a Jetson Nano platform for increased computational power. This will involve converting the ONNX model to the TensorRT format and using DeepStream for accelerated inference and improved response times.
- Incorporating reinforcement learning and robotic arm technology to develop an automated feature for playing Chinese chess, thus combining advanced AI strategies with physical interaction capabilities.
