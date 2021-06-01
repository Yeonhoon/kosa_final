import cx_Oracle

orcl_dsn = cx_Oracle.makedsn(host='localhost', port=1521, sid='orcl')
conn = cx_Oracle.connect(dsn = orcl_dsn, user='jyhoon94', password='Zpflrjs94')

class User:
    def __init__(self, user_id, user_email, user_pw, user_name):
        self.user_id = user_id
        self.user_email = user_email
        self.user_pw = user_pw
        self.user_name = user_name

    def signUp(self, user_id, user_email, user_pw, user_name):
        cursor = conn.cursor()
        sql = """insert into user_info values 
                (:user_id, :user_email, :user_pw, :user_name)"""
        cursor.execute(sql, (user_id, user_email, user_pw, user_name))
        conn.commit()