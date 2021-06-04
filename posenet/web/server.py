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


@app.route('/signup', methods=['POST'])
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
    print(x)
    # id = x[0][0]
    # sets = x[0][1]
    # reps = x[0][2]
    # dates = x[0][3]
    con = um.conn
    sql = """select TO_CHAR(squat_date, 'YYYY-MM-DD') as dates, user_id, sum(set_count * rep_count) as volume from squat_archive 
            where user_id= """ + "'" +session['user_id'] + "'" + "group by TO_CHAR(squat_date, 'YYYY-MM-DD'), user_id"
    df = pd.read_sql(sql, con=conn)
    fig = px.bar(df, x='DATES', y= 'VOLUME', template='plotly_white', color="VOLUME", 
                    color_continuous_scale='Teal')
    fig.update_xaxes(type="date")
    fig.update_layout(
            font_family = "Droid Sans",
            height = 330
            )
    graphJSON  = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    knee_query = """select user_id, set_count, rep_count, squat_date from squat_archive where user_id = '{}'""".format(session['user_id'])
    knee_df = pd.read_sql(knee_query, con=con)
    # print("df: ", df)
    knee_fig = px.line(knee_df, x='SQUAT_DATE', y='REP_COUNT', template="simple_white") #barmode='group'
    knee_graphJSON = json.dumps(knee_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    #정해진 달 (6월) 스쿼트한 날 찾기
    return render_template('main.html', id = session['user_id'], graphJSON=graphJSON,test_graphJSON=knee_graphJSON)

# @app.route('/mypage')
# def my_page():
#     return render_template('mypage.html')


# @app.route('/test')
# def notdash():
#     return render_template('test.html')

@app.route('/squartdays',methods=['POST'])
def squartdays():
    year = request.form['year']
    month = request.form['month']
    con = um.conn
    days_query="""select DISTINCT to_char(squat_date,'DD') as day FROM squat_archive
                where user_id='{}' and to_char(squat_date,'YYYY')='{}'
                and to_char(squat_date,'MM')='{:02d}'
                """.format(session['user_id'],year,int(month))
    days_df = pd.read_sql(days_query, con=con)
    days_list=list(map(int,days_df["DAY"].tolist()))
    days=",".join(map(str,days_list))
    

    return jsonify(result=days)

@app.route('/kneeAngleGraph',methods=['POST'])
def kneeAngleGraph():
    year = request.form['year']
    month = request.form['month']
    date = request.form['date']
    if date[-1]=="★":
        date=date[:-1]
    con = um.conn
    knee_query = """select angles, frame from knee_angle
                    where user_id='{}' and to_char(angle_date,'YYYY-MM-DD')='{}-{:02d}-{:02d}'
                    ORDER BY frame""".format(session['user_id'],year,int(month),int(date))
    knee_df = pd.read_sql(knee_query, con=con)
    knee_fig = px.line(knee_df, x='FRAME', y='ANGLES', template="simple_white") #barmode='group'
    knee_graphJSON = json.dumps(knee_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return knee_graphJSON


if __name__ == '__main__':
    app.run(debug=True)


    