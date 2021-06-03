import cx_Oracle
from flask_login import UserMixin

orcl_dsn = cx_Oracle.makedsn(host='localhost', port=1521, sid='orcl')
conn = cx_Oracle.connect(dsn = orcl_dsn, user='jyhoon94', password='123')

class User:
    # def __init__(self, user_id, user_email, user_pw, user_name):
    #     self.user_id = user_id
    #     self.user_email = user_email
    #     self.user_pw = user_pw
    #     self.user_name = user_name

    def signUp(self, user_id, user_email, user_pw, user_name):
        cursor = conn.cursor()
        sql = """insert into user_info values 
                (:user_id, :user_email, :user_pw, :user_name)"""
        cursor.execute(sql, (user_id, user_email, user_pw, user_name))
        conn.commit()
    
    def checkId(self, user_id):
        cursor = conn.cursor()
        sql = """
            select user_id from user_info where user_id=:user_id
            """
        cursor.execute(sql, {"user_id":user_id})
        result = cursor.fetchone()
        return result

    def checkPw(self, user_id):
        cursor = conn.cursor()
        sql = """
            select user_pw from user_info where user_id=:user_id
        """
        cursor.execute(sql, {"user_id":user_id})
        result = cursor.fetchone()
        return result

    def login(self, user_id, user_pw):
        cursor = conn.cursor()
        sql = "select user_id, user_pw from user_info where user_id=:user_id and user_pw=:user_pw"
        cursor.execute(sql, {"user_id":user_id, "user_pw": user_pw})
        result = cursor.fetchone()
        return result
    
    def get_squat_data(self, user_id):
        cursor = conn.cursor()
        sql = "select user_id, set_count, rep_count, squat_date from squat_archive where user_id=:user_id"
        cursor.execute(sql, {"user_id":user_id})
        x = cursor.fetchall()
        return x
    
        