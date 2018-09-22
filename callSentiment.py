import ticketSentiment as s
import PyMySQL

db = PyMySQL.connect("localhost","maian","Gegwegpw821!!!@oj12","maian")

cursor = db.cursor()


cursor.execute("select comments from msp_tickets order by rand() limit 1")

data = cursor.fetchone()

print(data)

print(s.sentiment(data))
