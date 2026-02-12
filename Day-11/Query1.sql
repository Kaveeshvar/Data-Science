CREATE TABLE emp_details (
    Name VARCHAR(25),
    Age int,
    sex CHAR(1),
    DOJ DATE,
    City varchar(15),
    salary float
);
-- DESCRIBE
PRAGMA table_info([emp_details]);
insert into emp_details
values("Kavee", 35, "M", "2005-05-30", "Chicago", 70000),
    ("Shane", 30, "M", "1999-06-25", "Seattle", 55000),
    ("Marry", 28, "F", "2009-03-10", "Boston", 62000),
    ("Dwayne", 37, "M", "2011-07-12", "Austin", 57000),
    ("Sara", 32, "F", "2017-10-27", "New York", 72000),
    ("Ammy", 35, "F", "2014-12-20", "Seattle", 80000);
CREATE TABLE Employees (
    Emp_Id INT PRIMARY KEY,
    Emp_name VARCHAR(50),
    Age INT,
    Gender CHAR(1),
    Doj DATE,
    Dept VARCHAR(50),
    City VARCHAR(50)
);
INSERT INTO Employees (Emp_Id, Emp_name, Age, Gender, Doj, Dept, City)
VALUES (
        104,
        'Dwayne',
        37,
        'M',
        '2011-07-12',
        'HR',
        'Austin'
    ),
    (
        105,
        'Sara',
        32,
        'F',
        '2017-10-27',
        'Finance',
        'New York'
    ),
    (
        106,
        'Ammy',
        35,
        'F',
        '2014-12-20',
        'Engineering',
        'Seattle'
    ),
    (
        107,
        'John',
        29,
        'M',
        '2010-08-15',
        'Sales',
        'Chicago'
    ),
    (
        108,
        'Emily',
        31,
        'F',
        '2012-11-05',
        'Marketing',
        'Seattle'
    ),
    (
        109,
        'Michael',
        33,
        'M',
        '2008-04-22',
        'Product',
        'Boston'
    ),
    (
        110,
        'Jessica',
        27,
        'F',
        '2015-09-30',
        'HR',
        'Austin'
    ),
    (
        111,
        'David',
        36,
        'M',
        '2006-02-14',
        'Finance',
        'New York'
    ),
    (
        112,
        'Sophia',
        34,
        'F',
        '2013-01-10',
        'Engineering',
        'Seattle'
    );
select *
from employees;
-- Add column Salary
ALTER TABLE Employees
ADD COLUMN Salary FLOAT;
-- Enter Salary details
UPDATE Employees
SET Salary = CASE
        WHEN Emp_Id = 101 THEN 57000
        WHEN Emp_Id = 102 THEN 72000
        WHEN Emp_Id = 103 THEN 80000
        WHEN Emp_Id = 104 THEN 57000
        WHEN Emp_Id = 105 THEN 72000
        WHEN Emp_Id = 106 THEN 80000
        WHEN Emp_Id = 107 THEN 55000
        WHEN Emp_Id = 108 THEN 62000
        WHEN Emp_Id = 109 THEN 70000
        WHEN Emp_Id = 110 THEN 48000
        WHEN Emp_Id = 111 THEN 75000
        WHEN Emp_Id = 112 THEN 68000
    END;

select emp_name,
    dept,
    salary
from employees
where salary > (
        select avg(salary)
        from employees
    );

    select emp_name, gender, dept, salary
from employees where salary >
(select salary from employees where emp_name = 'John');