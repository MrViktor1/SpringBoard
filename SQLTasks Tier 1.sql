/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 1 of the case study, which means that there'll be more guidance for you about how to 
setup your local SQLite connection in PART 2 of the case study. 

The questions in the case study are exactly the same as with Tier 2. 

PART 1: PHPMyAdmin
You will complete questions 1-9 below in the PHPMyAdmin interface. 
Log in by pasting the following URL into your browser, and
using the following Username and Password:

URL: https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

In this case study, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */


/* QUESTIONS 
/* Q1: Some of the facilities charge a fee to members, but some do not.
Write a SQL query to produce a list of the names of the facilities that do. */

    SELECT name 
    FROM facilities 
    WHERE membercost > 0
    ;

/* Q2: How many facilities do not charge a fee to members? */

    SELECT COUNT(name)
    FROM facilities 
    WHERE membercost = 0
    ;

/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

    SELECT name, facid, membercost, monthlymaintenance
    FROM facilities 
    WHERE membercost > 0
        AND membercost < .2 * monthlymaintenance
    ;

/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */

    SELECT *
    FROM facilities
    WHERE facid IN (1,5)
    ;

/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */

    SELECT name, monthlymaintenance, 
        CASE WHEN monthlymaintenance <= 100 THEN 'expensive'
            WHEN monthlymaintenance > 100 THEN 'cheap'
            END AS expense_label
    FROM facilities 
    ;

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */

    SELECT firstname, surname, MAX(joindate) AS most_recent_member
    FROM members
    ;
    
/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

    SELECT DISTINCT (firstname || ' ' || surname) AS member, name
    FROM members
    LEFT JOIN bookings
        ON members.memid = bookings.memid
    LEFT JOIN facilities
        ON bookings.facid = facilities.facid
    WHERE members.memid > 0
        AND name LIKE '%tennis court%'
    ORDER BY name
    ;

/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

    SELECT name, (firstname || ' ' || surname) AS member, 
        CASE WHEN b.memid > 0 THEN (slots * membercost)
            WHEN b.memid = 0 THEN (slots * guestcost) 
            END AS cost
    FROM bookings AS b
    LEFT JOIN members AS m
        ON b.memid = m.memid
    LEFT JOIN facilities AS f
        ON b.facid = f.facid
    WHERE cost > 30
        AND starttime like '%2012-09-14%'
    ORDER BY cost DESC
    ;

/* Q9: This time, produce the same result as in Q8, but using a subquery. */

    
    SELECT 
            (SELECT firstname || ' ' || surname 
            FROM members) AS member,
        CASE WHEN memid > 0 THEN (slots * membercost)
            WHEN memid = 0 THEN (slots * guestcost) 
            END AS cost
    FROM bookings,
        (SELECT membercost, guestcost
        FROM facilities)
    WHERE starttime like '%2012-09-14%'
     AND cost > 30
    ;

/* PART 2: SQLite
/* We now want you to jump over to a local instance of the database on your machine. 

Copy and paste the LocalSQLConnection.py script into an empty Jupyter notebook, and run it. 

Make sure that the SQLFiles folder containing thes files is in your working directory, and
that you haven't changed the name of the .db file from 'sqlite\db\pythonsqlite'.

You should see the output from the initial query 'SELECT * FROM FACILITIES'.

Complete the remaining tasks in the Jupyter interface. If you struggle, feel free to go back
to the PHPMyAdmin interface as and when you need to. 

You'll need to paste your query into value of the 'query1' variable and run the code block again to get an output.
 
QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

    SELECT name, (membercost * mem_slots + guestcost * guest_slots - monthlymaintenance) AS total_rev
    FROM facilities AS f
    LEFT JOIN (SELECT facid, SUM(slots) AS mem_slots
                FROM bookings
                WHERE memid > 0
                GROUP BY facid) AS sq1       
        ON f.facid = sq1.facid
    LEFT JOIN (SELECT facid, SUM(slots) AS guest_slots
                FROM bookings
                WHERE memid = 0
                GROUP BY facid) AS sq2       
        ON f.facid = sq2.facid
    WHERE total_rev > 1000
    ;

/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */
    
    SELECT o.surname || ', ' || o.firstname AS member, s.surname || ', ' || s.firstname AS recommended_by
    FROM members AS o, members AS s
    WHERE o.recommendedby = s.memid
    ORDER BY member 
    ;

/* Q12: Find the facilities with their usage by member, but not guests */
    
    SELECT name, 
           COUNT(slots) AS member_usage
        FROM bookings as b
            LEFT JOIN facilities AS f
                ON b.facid = f.facid
        WHERE memid > 0
        GROUP BY name
         ;

/* Q13: Find the facilities usage by month, but not guests */
    
    SELECT name, 
           sum(slots),
           CASE WHEN starttime LIKE '%-01-%' THEN 'JAN'
                WHEN starttime LIKE '%-02-%' THEN 'FEB'
                WHEN starttime LIKE '%-03-%' THEN 'MAR'
                WHEN starttime LIKE '%-04-%' THEN 'APR'
                WHEN starttime LIKE '%-05-%' THEN 'MAY'
                WHEN starttime LIKE '%-06-%' THEN 'JUN'
                WHEN starttime LIKE '%-07-%' THEN 'JUL'
                WHEN starttime LIKE '%-08-%' THEN 'AUG'
                WHEN starttime LIKE '%-09-%' THEN 'SEP'
                WHEN starttime LIKE '%-10-%' THEN 'OCT'
                WHEN starttime LIKE '%-11-%' THEN 'NOV'
                WHEN starttime LIKE '%-12-%' THEN 'DEC'
                END AS month
        FROM bookings as b
            LEFT JOIN facilities AS f
                ON b.facid = f.facid
        WHERE memid > 0 
        GROUP BY name, month
         ;
