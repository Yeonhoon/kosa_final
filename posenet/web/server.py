# coding: utf-8
from re import template
from flask import Flask, jsonify, render_template, request, session, redirect
import pandas as pd
import user_mgmt as um
import os
from flask_login import LoginManager
import plotly
import plotly.express as px
import json

user = um.User()

# from control import user_mgmt as um
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Login Session
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return user.checkId(user_id)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/signupform')
def signUpform():
    return render_template('user/signup.html')

@app.route('/checkMid', methods=['POST'])
def checkId():
    data = request.form['check_id'] # 키로 받아야 함. 그래야 value를 사용 가능
    if user.checkId(data) is None:
        return jsonify(result='success')
    else:
        return jsonify(result='failed')

@app.route('/signup', methods=['POST'])
def signUp():
    user_id = request.form['mid']
    user_email = request.form['memail']
    user_pw = request.form['mpw']
    user_name = request.form['mname']
    user.signUp(user_id, user_email, user_pw, user_name)
    return redirect('/')

@app.route('/loginform', methods=['GET'])
def loginForm():
    return render_template('user/login.html')

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form['id']
    user_pw = request.form['password']
    result = user.login(user_id, user_pw)
    try:
        print(user.checkId(user_id)[0])
    except:
        print('등록되지 않은 아이디')
    else:
        session['user_id'] = result[0]  
        return redirect('/')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route('/main')
def dash_page():
    conn = um.conn
    # 'select * from squat_archive' + " where user_id= '"
    sql = """select TO_CHAR(squat_date, 'YYYY-MM-DD') as dates, user_id, sum(set_count * rep_count) as count from squat_archive 
            where user_id= """ + "'" +session['user_id'] + "'" + "group by TO_CHAR(squat_date, 'YYYY-MM-DD'), user_id"
    df = pd.read_sql(sql, con=conn)
    print(df)
    # df['TOTAL'] = df['SET_COUNT'] * df['REP_COUNT']
    fig = px.bar(df, x='DATES', y= 'COUNT', template='simple_white')
    fig.update_xaxes(type="date")
    fig.update_layout(
            autosize=True,
            font_family = "Droid Sans",
            width = 67, height = 33
            )
    graphJSON  = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('main.html', id = session['user_id'], graphJSON=graphJSON )

# @app.route('/mypage')
# def my_page():
#     return render_template('mypage.html')

# @app.route('/main')
# def notdash():
#     con = um.conn
#     cursor = con.cursor()
#     query = """select user_id, set_count, rep_count, squat_date from squat_archive where user_id=:user_id"""
#     df = pd.read_sql(query, con=con)
#     print("df: ", df)
#     fig = px.bar(df, x='SET_COUNT', y='REP_COUNT', color='SQUAT_DATE',
#             template="simple_white") #barmode='group'
#     graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#     # print(graphJSON)
#     return render_template('test.html', graphJSON=graphJSON)


if __name__ == '__main__':
    app.run(debug=True)
    