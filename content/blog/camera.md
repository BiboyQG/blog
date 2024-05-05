+++
title = 'IoT-Enabled Home Security Camera'
date = 2024-05-05T16:18:51-05:00
draft = false
categories = ['Internet of Things']
tags = ['Project', 'Application', 'Internet of Things']
+++

[Video][https://www.youtube.com/watch?v=Pth99K0GmAg] [GitHub][https://github.com/IoT-God]

#### 1. Motivation

Facial recognition technology has become prevalent in all areas of life. Whether you work in security, law enforcement, or manufacture personal devices, the presence of facial recognition for various purposes is evident. Our project seeks to dive into this increasingly common technology and apply it to a place that needs upgrades, such as banks. Many banks are on old applications or using outdated technology, limiting the effectiveness of their work. Whether it is a concern over low-quality security cameras, which make it hard to review footage afterwards, or a 20-year-old system used for working every day, many aspects of banks can be improved upon with simple computer vision implementations. For logging into work, a secure application allowing for user authentication is an easy starting point. When logging into a social media account, a website, or even a phone, one easy way people prefer to verify their identity is using facial recognition.

However, implementing a barebones facial recognition system is problematic. While facial recognition provides a sense of security in regards to the fact that it is not something that can be physically stolen, like a credit card or key to an account, it faces other concerns in terms of its accuracy and efficiency when granting access to a user; What if this facial recognition was not effective? On a home security system, an intruder could unlock a door and invade one’s home or take their belongings. Alternatively, people could pretend to be someone else and gain access to personal information or accounts, with unimaginable consequences. One can imagine the horror of a poor model allowing even slightly similar faces to be verified, or for extremely poor models, potentially even a picture or cutout of the account owner’s face. For this reason, many put their trust into those creating these systems, hoping for the most efficient facial recognition.

Our project aims to meet the hopes of those using facial recognition, creating an efficient and consistent facial recognition program that can store and retrieve facial data to be used whenever needed. Using machine learning and training on datasets, we created an accurate facial recognition system that will avoid false positives and allow only the appropriate users access to their most personal information. In addition, we leverage the flexibility of such models to recognize information such as body language/position and license plate text, which come in handy identifying suspicious individuals and recording information about those who have committed a crime such as robbery at a bank, who may get away before physical intervention can occur. Overall, developing a program utilizing a camera for facial recognition and other computer vision provides opportunities to address security challenges, enhance user experiences, and contribute to technological advancements across various industries.

#### 2. Technical Approach

![img](https://s2.loli.net/2024/05/06/NCoqD9iTL7RVx4z.png)

In designing the technical approach for this system, we adopted a modular architecture consisting of a frontend using React, two backends implemented by Flask and SpringBoot, and a Raspberry Pi acting as the hardware intermediary. The React frontend provides the user interface, enabling interaction with the system, while the Flask backend handles server-side detecting algorithm and detection results requests with SpringBoot dealing with user operation API. The Raspberry Pi serves as the bridge between the software and the hardware, facilitating image capture from the camera module and executing image processing tasks using OpenCV. The data flow within the system follows a structured path: the Raspberry Pi continuously captures images, which are then processed using the Histogram of Oriented Gradients (HOG) algorithm for object recognition and facial identification. The processed data is subsequently transmitted to the Flask backend for storage or further analysis. Finally, the React frontend communicates with the Flask backend to request processed data and display it to the user, completing the data flow loop.

#### 3. Implementation Details

In implementing this system, a combination of software packages, hardware modules, algorithms, and data structures were utilized. The software stack includes React for frontend development, Flask and SpringBoot for backend server implementation, OpenCV for image processing tasks on the Raspberry Pi, and HOG for machine learning-based tasks.

Hardware components essential to the system include a Raspberry Pi and a camera for image capture. Additionally, a remote server which has public IPv4 ip address are critical for the owner of the system to get access to the system no matter where he/she is.

Data transfer between components was facilitated using TCP and tailscale to accomplish LAN access, while Pickle was employed for serializing and deserializing complex Python objects for storage. This comprehensive implementation ensured the system's efficiency and accuracy in real-time image processing and recognition tasks.

The main datasets, or their libraries for code modeling and importing functions, were as follows:

- https://www.tomshardware.com/how-to/raspberry-pi-facial-recognition for facial recognition
- https://github.com/pm1715/HAR_Project/tree/main for actions based on body position
- https://github.com/opencv/opencv/tree/master/data/haarcascades, pytesseract for facial recognition, license plate reading
- General libraries and related datasets for React, Flask, SpringBoot, OpenCV

The overall structure of our application can be found above.
