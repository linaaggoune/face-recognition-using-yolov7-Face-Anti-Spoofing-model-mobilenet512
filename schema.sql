create table users( id integer primary key AUTOINCREMENT, name text not null, password text not null, admin boolean not null DEFAULT '0');


create table emp ( empid integer primary key AUTOINCREMENT, name text not null, email text, phone integer, address text, joining_date timestamp DEFAULT CURRENT_TIMESTAMP,  total_projects integer default 1, total_test_case integer DEFAULT 1, total_defects_found integer DEFAULT 1, total_defects_pending integer DEFAULT 1 );

CREATE TABLE meeting (
  meetid INTEGER PRIMARY KEY,
  meet_title TEXT,
  user_id INTEGER,
  employee_ids TEXT,
  date_of_meeting DATE,
  start_time TIME,
  end_time TIME,
  meeting_place TEXT,
  order_of_the_day TEXT,
  FOREIGN KEY (user_id) REFERENCES users (id),
  FOREIGN KEY (employee_ids) REFERENCES emp (empid)
);
