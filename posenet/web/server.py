# coding: utf-8
from flask import Flask, jsonify, render_template, request, session, redirect
import pandas as pd
import user_mgmt
# from control import user_mgmt as um
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/signupform')
def signUpform():
    return render_template('user/signup.html')

@app.route('/signup', methods=['POST'])
def signUp():
    user_id = request.form('mid')
    user_email = request.form('mmeail')
    user_pw = request.form('mpw')
    user_name = request.form('mname')
    user_mgmt.signup(user_id, user_email, user_pw, user_name)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)