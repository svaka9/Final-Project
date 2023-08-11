**Group:**

1. Sahasraa Vaka (801327852)
2. Hari Chandana Pulaparti (801329190)
3.Abhilash Bandi (801329252)
4. Subramanya Lalitha Adithya Prasad Gopaluni Venkata (801328236)

**Project Title:**

Full body generation using WebRTC protocol.

**Objective:**

This project aims to automatically generate whole body pose animation on 2D Character model based on the camera input. It uses Pose estimation models to create entire body animation on 2D virtual character.

**Installation Guide:**

System Requirements:

1. OS: Windows/IOS
2. A device with a camera: The user's device must have a camera to capture video of their body.

Software Requirements:

- Python
- opencv-python-headless 3.4.2
- mediapipe 0.8.6
- streamlit\_webrtc 0.44.2

How to run the application:

Step 1: Install the required dependencies using the following command

Command: **pip install -r requirements.txt** 

Step 2: Once the dependencies are installed, now run the application using the command below

Command: **streamlit run streamlit\_app.py** 

After running the above command, you can view the application at [_ **http://localhost:8501/** _](http://localhost:8501/)

**UI:**

![image](https://user-images.githubusercontent.com/124633158/235562791-0126dc8e-949f-4132-a218-b1836e833424.png)

Above figure shows the UI of the application with different options like pictogram, pose and both. Choosing pictogram would show the 2d model of the human with pose animation, Choosing pose would show the raw video of the user and finally choosing both option would show the user, raw input in column one  and the 2d character with the same pose as human in column two.


![image](https://user-images.githubusercontent.com/124633158/235562818-9a426301-30fc-4268-bc61-192b9cca45c8.png)

Above figure shows the sample instance of the application output when the user chooses the both option.

