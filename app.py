import csv
from flask import Flask, url_for, request, session, g
from flask.templating import render_template
from matplotlib import pyplot as plt 
from werkzeug.utils import redirect
from database import get_database
from werkzeug.security import generate_password_hash, check_password_hash
import os
import sqlite3
import json
import cv2
import numpy as np
from deepface import DeepFace
import torch
from torchvision.transforms import transforms
from PIL import Image
import json
import base64
from io import BytesIO
from datetime import date
from datetime import datetime
import pandas as pd
from flask import render_template, redirect, url_for
from keras.models import model_from_json
from flask import Flask, render_template
from flask import Flask, render_template, request, redirect, url_for
from collections import defaultdict



app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)


@app.teardown_appcontext
def close_database(error):
    if hasattr(g, 'crudapplication_db'):
        g.crudapplication_db.close()

def get_current_user():
    user = None
    if 'user' in session:
        user = session['user']
        db = get_database()
        user_cur = db.execute('select * from users where name = ?', [user])
        user = user_cur.fetchone()
    return user


@app.route('/')
def index():
    user = get_current_user()
  
    return render_template('home.html', user = user)
@app.route('/')
def index1():
    user = get_current_user()
    return render_template('logout.html', user = user)

@app.route('/login', methods = ["POST", "GET"])
def login():
    user = get_current_user()
    error = None
    db = get_database()
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        user_cursor = db.execute('select * from users where name = ?', [name])
        user = user_cursor.fetchone()
        if user:
            if check_password_hash(user['password'], password):
                session['user'] = user['name']
                return redirect(url_for('dashboard'))
            else:
                error = "Username or Password did not match, Try again."
        else:
            error = 'Username or password did not match, Try again.'
    return render_template('lolo.html', loginerror = error, user = user)

@app.route('/register', methods=["POST", "GET"])
def register():
    user = get_current_user()
    db = get_database()
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        dbuser_cur = db.execute('select * from users where name = ?', [name])
        existing_username = dbuser_cur.fetchone()
        if existing_username:
            return render_template('register.html', registererror = 'Username already taken , try different username.')
        db.execute('insert into users ( name, password) values (?, ?)',[name, hashed_password])
        db.commit()
        return redirect(url_for('index'))
    return render_template('register.html', user = user)

@app.route('/dashboard')
def dashboard():
    user = get_current_user()
    db = get_database()
    emp_cur = db.execute('select * from emp')
    allemp = emp_cur.fetchall()
    
    return render_template('dashiiibord.html', user = user, allemp = allemp)

@app.route('/addnewemployee', methods = ["POST", "GET"])
def addnewemployee():
    user = get_current_user()
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        db = get_database()
        db.execute('insert into emp (name, email, phone ,address) values (?,?,?,?)', [name, email, phone, address])
        db.commit()
        return redirect(url_for('dashboard'))
    return render_template('addemp.html', user = user)

@app.route('/meetings', methods=['GET', 'POST'])
def meetings():
    conn = sqlite3.connect('crudapplication.db')
    cursor = conn.cursor()
    
    # Fetch data from the database
    cursor.execute("SELECT m.meetid, m.meet_title, u.name AS coordinator FROM meeting m INNER JOIN users u where m.user_id=u.id ")
    meetings = cursor.fetchall()
    
    conn.close()
    
    return render_template('meetinggs.html', meetings=meetings)
 #he works good
@app.route('/meetings/<int:meetid>')
def singlemeeting(meetid):
    session_user = get_current_user()
    conn = sqlite3.connect('crudapplication.db')
    cursor = conn.cursor()

    # Retrieve meeting details from the database
    cursor.execute("SELECT * FROM meeting INNER JOIN users u ON user_id = u.id WHERE meetid = ?", [meetid])
    meeting = cursor.fetchone()

    if meeting:
        meet_dict = {
            "meetid": meeting[0],
            "meet_title": meeting[1],
            "user_id": meeting[2],
            "date_of_meeting": meeting[3],
            "start_time": meeting[4],
            "end_time": meeting[5],
            "meeting_place": meeting[6],
            "order_of_the_day": meeting[7]
        }

        # Retrieve user details
        cursor.execute("SELECT name FROM users WHERE id = ?", [meet_dict["user_id"]])
        user = cursor.fetchone()
        meet_dict["user_id"] = user[0]

        # Retrieve employee details
        cursor.execute("SELECT e.empid, e.name FROM emp e INNER JOIN resultattendance ra ON e.email = ra.identi WHERE ra.meeting_id = ?", [meetid])
        employees = cursor.fetchall()
        emp_list = []
        present_employees = set()
        for emp in employees:
            present_employees.add(emp[1])  # Add present employee name to the set
            emp_dict = {
                "empid": emp[0],
                "name": emp[1],
                "presence": True  # Set presence status to True for present employees
            }
            emp_list.append(emp_dict)

        # Retrieve all employees
        cursor.execute ("SELECT * FROM emp INNER JOIN attendance a ON emp.empid = a.empid WHERE a.meetid = ?", [meetid] )
        all_employees = cursor.fetchall()
        for emp in all_employees:
            if emp[1] not in present_employees:
                emp_dict = {
                    "empid": emp[0],
                    "name": emp[1],
                    "presence": False  # Set presence status to False for absent employees
                }
                emp_list.append(emp_dict)

        meet_dict["employee_ids"] = emp_list

        return render_template('singlmeet.html', user=session_user, meeting=meet_dict)
    else:
        return "Meeting not found"


@app.route('/deletemeet/<int:meetid>', methods = ["GET", "POST"])
def deletemeet(meetid):
    user = get_current_user()
    if request.method == 'GET':
        db = get_database()
        db.execute('delete from meeting where meetid = ?', [meetid])
        db.commit()
        return redirect(url_for('meetings'))
    return render_template('meetinggs.html', user = user)



@app.route('/singleemployee/<int:empid>')
def singleemployee(empid):
    user = get_current_user()
    db = get_database()
    emp_cur = db.execute('select * from emp where empid = ?', [empid])
    single_emp = emp_cur.fetchone()
    return render_template('singleemp.html', user = user, single_emp = single_emp)

@app.route('/fetchone/<int:empid>')
def fetchone(empid):
    user = get_current_user()
    db = get_database()
    emp_cur = db.execute('select * from emp where empid = ?', [empid])
    single_emp = emp_cur.fetchone()
    return render_template('updatemp.html', user = user, single_emp = single_emp)

@app.route('/updateemployee' , methods = ["POST", "GET"])
def updateemployee():
    user = get_current_user()
    if request.method == 'POST':
        empid = request.form['empid']
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        db = get_database()
        db.execute('update emp set name = ?, email =? , phone = ? , address = ? where empid = ?', [name, email, phone, address, empid])
        db.commit()
        return redirect(url_for('dashboard'))
    return render_template('updatemp.html', user = user)

@app.route('/deleteemp/<int:empid>', methods = ["GET", "POST"])
def deleteemp(empid):
    user = get_current_user()
    if request.method == 'GET':
        db = get_database()
        db.execute('delete from emp where empid = ?', [empid])
        db.commit()
        return redirect(url_for('dashboard'))
    return render_template('dashboard.html', user = user)

   

    # Load the first model: YOLOv7 for face detection
model01 = torch.hub.load('C:\\Users\\amgsoft\\Downloads\\vs_code_yolo\\yolov7', 'custom', 'C:\\Users\\amgsoft\\Downloads\\vs_code_yolo\\yolov7\\best (4).pt', source='local', force_reload=True)

with open('yolov7\\antispoofing_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model architecture from JSON
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights('yolov7\\antispoofing_model.h5')
def preprocess_face(image):
    # Resize the image to the required input shape of the model
    input_shape = (160, 160)
    resized = cv2.resize(image, input_shape)

    # Convert the image to RGB and normalize the pixel values
    normalized = resized / 255.0

    # Add an extra dimension to match the model's input shape
    preprocessed = np.expand_dims(normalized, axis=0)

    return preprocessed
df_identities = None
@app.route('/modelPredict', methods=["POST", "GET"])
def modelPredict():

    global df_identities
    

    json_data = request.get_json()

    # Extract the frame data from the JSON object
    frame_data = json_data.get('frame').split(",")[1]

    # Decode the base64 encoded frame data
    decoded_data = base64.b64decode(frame_data)

    # Create a PIL Image object from the decoded data
    image = Image.open(BytesIO(decoded_data))
    image = image.convert('RGB')

    # Convert the PIL Image to OpenCV format (numpy array)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model01(frame)
    df = results.pandas().xyxy[0]



    for _, row in df.iterrows():
        if row['class'] == 0 and row['confidence'] > 0.5:
            # Extract the bounding box coordinates
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])

            # Draw a rectangle around the face
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Extract the face region as a separate image
            face_img = frame[ymin:ymax, xmin:xmax]
            preprocessed_face = preprocess_face(face_img)

            # Make a prediction using the loaded anti-spoofing model
            prediction = model.predict(preprocessed_face)

            # Classify the prediction as real or spoof based on a threshold
            threshold = 0.5  # Adjust the threshold as per your model and requirements
            if prediction > threshold:
                # Face recognition
                print('It is a real face.')
                dfs = DeepFace.find(img_path=face_img, enforce_detection=False, db_path="C:\\Users\\amgsoft\\Downloads\\vs_code_yolo\\pic\\linaaggoun", model_name='Facenet512')
                #print(dfs)
                identities = [result["identity"] for result in dfs]

                    # Print the identities
                for identity in identities:
                        print(identity)
                        
                        #stock all the identities on dataframe
                df_identities = pd.DataFrame({"identity": [result["identity"] for result in dfs]})
               
                # Assuming you have a DataFrame called 'result_df'
                df_identities.to_csv('temp.csv', index=False)
                # Read the CSV file
                df = pd.read_csv('temp.csv')
                

                print(df_identities) 
                 
            else:
                print('It is a fake face.')
    return ""

@app.route('/start_meeting/<int:meetid>', methods=["POST"])
def start_meeting(meetid):
    user = get_current_user()
    return render_template('runningmeeting.html', user=user)



@app.route('/rapport_meeting/<int:meetid>', methods=["POST"])
def rapport_meeting(meetid):
    user = get_current_user()
    data = pd.read_csv('resultrealtime.csv')
    identities = data['identity'].str.split('\n').tolist()[0]
    

# Remove the prefixes from each line
    identities = [line.split(' ', 1)[-1] for line in identities]
    identities = [line.replace('imgs', '') for line in identities]
    identities = [line.replace('\\', '') for line in identities]
    

    # Connect to the database
    conn = sqlite3.connect('crudapplication.db')
    cursor = conn.cursor()

    # Iterate over the identities and insert them into the "resultattendance" table
    # Now you can access each line separately
    for identity in identities:
        print('aaaaaaaaaaaaaaaa')
        print(identity)
        print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
        # Insert the identity into the "resultattendance" table
        cursor.execute("INSERT INTO resultattendance (identi, meeting_id) VALUES (?, ?)", (identity, meetid))
        print('finish')
    # Commit the changes
        conn.commit()

    # Close the database connection
    #conn.close()

    return render_template('rapport.html', user=user, identities=identities)


@app.route('/end_meeting/<int:meetid>', methods=["POST"])
def end_meeting(meetid):
   

    return render_template('singlmeet.html')
 
@app.route('/statistiques')
def statistiques():
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    db = get_database()
    cursor = db.execute('''SELECT meeting.meet_title, COUNT(attendance.empid) AS meeting_count
                        FROM meeting LEFT JOIN attendance ON meeting.meetid = attendance.meetid 
                        ''')
    result = cursor.fetchall()
    stat = []
    for row in result:
        stat.append({
            'name' : row[0] , # Accessing by index
            'meeting_count' : row['meeting_count']  # Accessing by column name 
        })
        
        
       
    print(stat)
    return render_template('statistiques.html',stat=stat )






@app.route('/attendance')
def attendance():
    # Connect to the SQLite database
    conn = sqlite3.connect('crudapplication.db')
    c = conn.cursor()

    # Fetch the attendance data
    c.execute("""
        SELECT e.name, COUNT(ra.identi) AS presence_count
        FROM emp e
        LEFT JOIN resultattendance ra ON e.name = ra.identi
        GROUP BY e.name
    """)
    attendance_data = c.fetchall()

    # Close the database connection
    conn.close()

    # Extract the employer names and presence counts
    employers = [row[0] for row in attendance_data]
    presence_counts = [row[1] for row in attendance_data]

    # Create a bar chart using Matplotlib
    plt.bar(employers, presence_counts)
    plt.xlabel('Employer')
    plt.ylabel('Presence Count')
    plt.title('Employer Attendance')

    # Save the chart as an image
    chart_path = 'static/chart.png'  # Specify the path where you want to save the chart image
    plt.savefig(chart_path)

    # Render the HTML template with the attendance data and the chart image path
    return render_template('chart.html', attendance_data=attendance_data, chart_path=chart_path)
 
# @app.route('/absence')
# def absence():
#     # Connect to the SQLite database
#     conn = sqlite3.connect('crudapplication.db')
#     c = conn.cursor()

#     # Fetch the list of employees that should attend the meetings
#     c.execute("SELECT empid, name FROM emp")
#     employees = c.fetchall()

#     # Retrieve the absence count for each employee
#     absence_data = []
#     for emp in employees:
#         emp_id = emp[0]
#         emp_name = emp[1]

#         # Count the absence by checking if the employee's name is not present in resultattendance table
#         c.execute("SELECT COUNT(*) FROM resultattendance WHERE identi != ?", [emp_name])
#         absence_count = c.fetchone()[0]

#         absence_data.append((emp_name, absence_count))

#     # Close the database connection
#     conn.close()

#     # Render the HTML template with the absence data
#     return render_template('absence.html', absence_data=absence_data)


@app.route('/absence')
def absence():
    # Connect to the SQLite database
    conn = sqlite3.connect('crudapplication.db')
    c = conn.cursor()

    # Retrieve the list of employee names
    c.execute("SELECT name FROM emp")
    employee_names = [row[0] for row in c.fetchall()]

    # Retrieve the absence count for each employee
    absence_count = []
    for name in employee_names:
        # Check if the name exists in the resultattendance table
        c.execute("SELECT COUNT(*) FROM resultattendance WHERE identi != ?", [name])
        result = c.fetchone()

        if result is None:
            # Employee is absent
            absence_count.append(1)
            
        else:
            # Employee is present
            print("yyyyyyyyyyyyyyyyyyyyyyyy")

    # Close the database connection
    conn.close()

    # Render the HTML template with the absence data
    return render_template('absence.html', employee_names=employee_names, absence_count=absence_count)

@app.route('/logout')
def logout():
    session.pop('user', None)
    render_template('logout.html')
from flask import Flask, render_template, request, redirect, url_for

@app.route('/addmeet', methods=['GET', 'POST'])
def addmeet():
    user = get_current_user()
    if user is None:
        return redirect(url_for('login'))
    if request.method == 'POST':
        meet_title = request.form.get('meet_title')
        employee_ids = request.form.getlist('employee_ids')
        date_of_meeting = request.form.get('date_of_meeting')
        start_time = request.form.get('start_time')
        end_time = request.form.get('end_time')
        meeting_place = request.form.get('meeting_place')
        order_of_the_day = request.form.get('order_of_the_day')

        if not employee_ids:
            raise ValueError('Please select at least one employee')

        conn = sqlite3.connect('crudapplication.db')
        db = conn.cursor()
        db.execute('INSERT INTO meeting (meet_title, user_id, date_of_meeting, start_time, end_time, meeting_place, order_of_the_day) VALUES (?, ?, ?, ?, ?, ?, ?)',
                  [meet_title, user['id'], date_of_meeting, start_time, end_time, meeting_place, order_of_the_day])
        conn.commit()

        meeting_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        for employee_id in employee_ids:
            db.execute('INSERT INTO attendance (empid, meetid) VALUES (?, ?)', (employee_id, meeting_id))
        conn.commit()

        conn.close()

        return redirect(url_for('meetings'))

    conn = sqlite3.connect('crudapplication.db')
    conn.row_factory = sqlite3.Row
    employees = conn.execute('SELECT * FROM emp').fetchall()
    conn.close()

    return render_template('addmeet.html', employees=employees)





if __name__ == '__main__':
    app.run(debug = True)