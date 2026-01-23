-- Create tables:

-- employees(id, name, dept_id, salary)
-- departments(dept_id, dept_name)
CREATE TABLE employees(
    id INT PRIMARY KEY,
    name VARCHAR(50),
    dept_id INT,
    Salary INT
);

CREATE TABLE departments(
    dept_id INT,
    dept_name VARCHAR(20),
    FOREIGN KEY(dept_id) REFERENCES employees(dept_id)
);

-- Insert:

-- 6 employees

-- 3 departments
INSERT INTO employees
VALUES
    (1, "Kavee", 101, 10000),
    (2, "Sriman", 101,12000),
    (3, "Anu", 102, 15000),
    (4, "Karti",103, 20000),
    (5, "Mohinth",102,17000),
    (6,"Sriram",103,17500)
;

SELECT * from employees;

INSERT INTO departments
VALUES
(101,"Finance"),
(102, "Delivery"),
(103, "HR");

SELECT * from departments;

-- INNER JOIN employees + departments

SELECT * FROM employees
INNER JOIN departments ON employees.dept_id=departments.dept_id;

-- LEFT JOIN (find employees without dept)

SELECT id,name,dept_name
FROM employees
LEFT JOIN departments ON employees.dept_id=departments.dept_id;
-- Right JOIN
SELECT dept_name,name
FROM departments
RIGHT JOIN employees ON employees.dept_id=departments.dept_id;

-- Average salary per department

SELECT dept_name, AVG(Salary) AS Avg_salary
FROM departments
RIGHT JOIN employees ON employees.dept_id=departments.dept_id
GROUP BY dept_name;

-- Department with highest average salary

SELECT dept_name, Max(Max_Avg_salary) FROM (
    SELECT dept_name, AVG(Salary) AS Max_Avg_salary
    FROM departments
    RIGHT JOIN employees ON employees.dept_id=departments.dept_id
    GROUP BY dept_name
    );

-- Employees earning above dept average (SUBQUERY)

SELECT * FROM employees e
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE dept_id=e.dept_id
);

-- Count employees per department

Select dept_id, COUNT(id) AS Employee_count
from employees
GROUP BY dept_id;

-- Highest paid employee per department

SELECT name AS Highest_paid_employee, MAX(salary), departments.dept_name
FROM employees
LEFT JOIN departments ON employees.dept_id=departments.dept_id
GROUP BY employees.dept_id;