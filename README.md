# Attendance Checking

Utilizing computer vision and machine learning algorithms to automate the process of attendance checking
</br>



https://github.com/thaileqt/attendance-checking/assets/75875212/d8192a6e-03f2-42f3-bf1f-def649436076


## Try it out

Clone this repository to your local machine using the following command
```bash
$ git clone https://github.com/thaileqt/attendance-checking.git
$ cd attendance-checking
```
Manually install and run the project (I'm using python 3.10.11)
```bash
$ pip install -r requirements.txt
$ streamlit run app.py
```
or run through Docker (currently **not working** due to camera problem)
```bash
$ docker-compose up
```
Navigate to `http://localhost:8501` to access the web interface.

## Todo
* Enhance face recognition accuracy (e.g develop better model).
* Use database to store data instead of csv file.

