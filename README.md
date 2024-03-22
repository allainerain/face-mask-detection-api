# face-mask-detection-api

## Project Summary 📝
This project is an extension of the project [Mask Patrol](https://github.com/cessbub/CS180-MaskPatrol), a face mask detection and social distancing web application. It provides a RESTful API endpoint for a pre-trained face mask detection model built using Keras and TFLite, implemented with Flask.

While I was trying to look for ways to deploy our project online, I came across the option of creating an API endpoint and calling it in the web app. But even then, TensorFlow was still needed, so getting it deployed for free still wasn't feasible (my bad). (You can see my failures in the deployments tab 😭😭)

Although this project may not fully resolve the deployment issues, feel free to use this! If you happen to know an easier way for me to deploy our object detection project, let me know. 😀
u
## How to run this repo ▶️

1. Git clone the repository
2. Run in terminal
   - `pip install -r requirements.txt`
3. Run in terminal `python api/app.py`

## Testing the API endpoint using Postman

1. Use your own photos or the photos in `sample_test_images`
2. Make a post request to the `/predict` endpoint. Make sure that the key is a File type with the name `file`.
![image](https://github.com/allainerain/face-mask-detection-api/assets/56602966/707f320d-27c9-48fb-a447-25e4a194fd23)

3. View the results! The results show both the Keras prediction and tflite prediction
![image](https://github.com/allainerain/face-mask-detection-api/assets/56602966/cbf69ca9-8a64-40ef-9092-7b2127ba6d51)
![image](https://github.com/allainerain/face-mask-detection-api/assets/56602966/1fc7218c-ee05-44b2-a1e2-b28bfb75f0f0)
