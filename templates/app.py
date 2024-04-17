from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, auth, db

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase app
cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://projectdhh-81b8f-default-rtdb.firebaseio.com/'
})
ref = db.reference()

# Routes
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

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

if __name__ == '__main__':
    app.run(debug=True)
