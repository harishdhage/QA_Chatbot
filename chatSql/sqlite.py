import sqlite3

# connect to DB
dbConnection = sqlite3.connect("student.db")

# create cursor object to insert to table
cursor = dbConnection.cursor()

## Create table
table_info = """
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""
# Table name should be case sensitive
check_table = "SELECT name FROM sqlite_master WHERE type='table' AND name='STUDENT';"

result = cursor.execute(check_table).fetchone()
#result = cursor.fetchone()
print(f"result : {result}")
if result:
    print("Student table is already existing!!!")
else:
    cursor.execute(table_info)
    print("Student table is created!!!")

#insert data to table
cursor.execute("insert into STUDENT values('Harish','AI Engineer','A','56')")
cursor.execute("insert into STUDENT values('Lamba','AI Engineer','C','71')")
cursor.execute("insert into STUDENT values('Pooja','Civil Engineer','D','65')")
cursor.execute("insert into STUDENT values('Obama','QA engineer','A','43')")

print("Display all data base records")
data = cursor.execute("select * from student")

for rw_data in data:
    print(rw_data)

#commit the changes in DB
dbConnection.commit()
dbConnection.close()