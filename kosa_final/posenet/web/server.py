# coding: utf-8
from re import T, template
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


@app.route('/signup', methods=['POST','GET'])
def signUp():
    user_id = request.form['mid']
    user_email = request.form['memail']
    user_pw = request.form['mpw']
    user_name = request.form['mname']
    user.signUp(user_id, user_email, user_pw, user_name)
    return redirect('/')

@app.route('/checkMid', methods=['POST'])
def checkId():
    data = request.form['check_id'] # 키로 받아야 함. 그래야 value를 사용 가능
    print(data)
    if user.checkId(data) is None:
        return jsonify(result='success')
    else:
        return jsonify(result='failed')

@app.route('/loginform', methods=['GET'])
def loginForm():
    return render_template('user/login.html')

@app.route('/login', methods=['POST'])
def login():
    # ajax를 통해 검사를 받아야 하므로 ajax에서 선언한 check_id와 check_pw로 받아야함
    user_id = request.form['check_id']
    user_pw = request.form['check_pw']
    result = user.login(user_id, user_pw)
    if user.checkId(user_id) is None:
        return jsonify(result='ID_Fail')

    if user.checkPw(user_id)[0] != user_pw:
        return jsonify(result='PW_Fail') 
    try:
        session['user_id'] = result[0]  
    except Exception as e:
        print(e.args, "아이디 혹은 비밀번호를 확인해주세요")
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route('/main')
def dash_page():
    x = user.get_squat_data(session['user_id'])
    con = um.conn
    sql = """select TO_CHAR(squat_date, 'YYYY-MM-DD') as dates, user_id, sum(set_count * rep_count) as volume from squat_archive 
            where user_id= """ + "'" +session['user_id'] + "'" + "group by TO_CHAR(squat_date, 'YYYY-MM-DD'), user_id " + \
            "order by dates"
    df = pd.read_sql(sql, con=con)
    
    xx = df.to_json(orient='columns')
    #정해진 달 (6월) 스쿼트한 날 찾기
    return render_template('main.html', id = session['user_id'], xx=xx )

# @app.route('/mypage')
# def my_page():
#     return render_template('mypage.html')

@app.route('/temp')
def temps():
    return render_template('temp.html')


@app.route('/test')
def notdash():
    x = user.get_squat_data(session['user_id'])

    con = um.conn
    sql = """select TO_CHAR(squat_date, 'YYYY-MM-DD') as dates, user_id, sum(set_count * rep_count) as volume from squat_archive 
            where user_id= """ + "'" +session['user_id'] + "'" + "group by TO_CHAR(squat_date, 'YYYY-MM-DD'), user_id"
    df = pd.read_sql(sql, con=con)
    xx = df.to_json(orient='columns')
    # test = df.to_json()
    # print(test)
    return render_template('test2.html', xx=xx)

@app.route('/squartdays',methods=['POST'])
def squartdays():
    year = request.form['year']
    month = request.form['month']
    con = um.conn
    days_query="""select to_char(squat_date,'DD') as day,set_count,rep_count FROM squat_archive
                where user_id='{}' and to_char(squat_date,'YYYY')='{}' and to_char(squat_date,'MM')='{:02d}'
                """.format(session['user_id'],year,int(month))
    days_df = pd.read_sql(days_query, con=con)
    days_newdict={}
    for _, r in days_df.iterrows():
        if days_newdict.get(int(r["DAY"])):
            days_newdict[int(r["DAY"])]="★"
        else:
            days_newdict[int(r["DAY"])]="{}:{}".format(r["SET_COUNT"],r["REP_COUNT"])
            

    return jsonify(result=days_newdict)

@app.route('/tableMaker',methods=['POST'])
def kneeAngleGraph():
    year = request.form['year']
    month = request.form['month']
    date = request.form['date']
    con = um.conn
    time_query = """select to_char(squat_date,'HH24:MI:SS') as time,set_count,rep_count FROM squat_archive
                where user_id='{}' and to_char(squat_date,'YYYY')='{}' and to_char(squat_date,'MM')='{:02d}' and to_char(squat_date,'DD')='{:02d}'
    """.format(session['user_id'],year,int(month),int(date))
    time_df = pd.read_sql(time_query, con=con)
    time_dict=time_df.to_dict('records')
    return jsonify(result=time_dict)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)


    