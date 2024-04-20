from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import firebase_admin
from firebase_admin import credentials, auth, db
# from num_recognition import recognize_sign_language

app = Flask(__name__)


cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://projectdhh-81b8f-default-rtdb.firebaseio.com/'
})
ref = db.reference()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.get_user_by_email(email)
            auth_user = auth.sign_in_with_email_and_password(email, password)
            return redirect(url_for('home'))
        except auth.AuthError as e:
            error_message = e.message
            return render_template('login.html', error=error_message)
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password == confirm_password:
            try:
                user = auth.create_user(email=email, password=password)
                uid = user.uid
                ref.child('users').child(uid).set({
                    'name': name,
                    'email': email
                })
                return redirect(url_for('home'))
            except auth.AuthError as e:
                error_message = e.message
                return render_template('register.html', error=error_message)
        else:
            error_message = "Passwords do not match"
            return render_template('register.html', error=error_message)
    return render_template('register.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/lesson')
def lesson():
    return render_template('lesson.html')

@app.route('/number')
def number():
    # recognized_text = recognize_sign_language()
    # return render_template('number.html', recognized_text=recognized_text)
    return render_template('number.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames():
#     import num_recognition
#     image = num_recognition.imageTest
#     camera = cv2.imshow("Hand Tracking", image)
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    #    <!-- <div class="col-lg-8  offset-lg-2">
    #                 <h3 class="mt-5">Live Streaming</h3>
    #                 <img src="{{ url_for('video_feed') }}" width="80%" height="80%">
    #             </div> -->

@app.route('/alphabet')
def alphabet():
    return render_template('alphabet.html')

@app.route('/words')
def words():
    return render_template('words.html')

@app.route('/play')
def play():
    return render_template('play.html')

@app.route('/Contact')
def contact():
    return render_template('Contact.html')


if __name__ == '__main__':
    app.run(debug=True)