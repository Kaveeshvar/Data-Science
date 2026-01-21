-- students(id, name, age, city)
CREATE TABLE  Students(
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT, 
    city VARCHAR[20] );

INSERT INTO Students 
VALUES 
(001, 'Kavee', 22, 'Sathy'), 
(002, 'Shahana', 21, 'Tirupur'), 
(003, 'Sriman', 21, 'Elur'), 
(004, 'Mohinht', 22, 'Karamadai'), 
(005, 'Karti', 22, 'Coimbatore');

Select * FROM Students;

-- marks(student_id, subject, score)
CREATE TABLE Marks(
    student_id INT,
    subject VARCHAR[10],
    score INT,
    FOREIGN KEY (student_id) REFERENCES Students(id)
);

-- DROP table Marks;

INSERT INTO Marks
VALUES
(001,"English",99),
(001,"Maths",100),
(001,"Computer Science",100);

-- DELETE from Marks;
-- SELECT * FROM Marks;

INSERT INTO Marks
VALUES
(005,"English",40),
(005,"Maths",99),
(005,"Computer Science",22);


SELECT * FROM Students
WHERE age >=22;

SELECT 
students.name,
AVG(marks.score) AS Avg_score 
FROM Marks
JOIN Students
ON Marks.student_id=Students.id
GROUP BY Students.name;

SELECT 
students.name,
marks.subject,
marks.score
FROM Marks
JOIN Students
ON Marks.student_id=Students.id
WHERE Marks.score >80;

SELECT
students.name,
marks.score
FROM Marks
INNER JOIN Students
ON Marks.student_id=Students.id


SELECT city,count(name) AS "Student count per city"
from Students
GROUP BY city;


SELECT max(avg_score)
FROM(
    SELECT AVG(score) as avg_score
    FROM Marks
    GROUP BY student_id
);
