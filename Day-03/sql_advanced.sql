CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    amount INT
);

CREATE TABLE customers(
    customer_id INT,
    name VARCHAR(50),
    city VARCHAR(20),
    FOREIGN KEY(customer_id) REFERENCES orders(order_id)
);

INSERT INTO ORDERS 
VALUES
    (1,101,10000),
    (2,101,15000),
    (3,102,7500),
    (4,103,1250),
    (5,103,27500),
    (6,104,100),
    (7,104,100),
    (8,104,100);

INSERT INTO customers
VALUES
    (101,'Kavee','Bengaluru'),
    (102,'Sriman','Trichy'),
    (103,'Anu','Tirupur'),
    (104,'Karti','Coimbatore');

SELECT * FROM orders
JOIN customers ON orders.customer_id=customers.customer_id;

-- Total amount spent by each customer
SELECT name,SUM(amount)
from orders o
JOIN customers c ON o.customer_id=c.customer_id
GROUP BY o.customer_id;

-- Customers with total spend > average spend
SELECT * from orders
WHERE amount> (
    SELECT Avg(amount) 
    FROM orders o
    WHERE o.customer_id=customer_id
);

-- City-wise total sales
SELECT city,SUM(amount)
from orders o
JOIN customers c ON o.customer_id=c.customer_id
GROUP BY o.customer_id;

-- Customer with highest single order
SELECT name,MAX(amount)
from orders o
JOIN customers c ON o.customer_id=c.customer_id;

-- Customers who never placed an order
INSERT INTO customers
VALUES
    (105,'Mohinth','Bengaluru');

SELECT c.*
FROM customers c
LEFT JOIN orders o
ON o.customer_id=c.customer_id
WHERE o.customer_id IS NULL;

-- Top 2 customers by total spend
SELECT name,SUM(amount)
from orders o
JOIN customers c ON o.customer_id=c.customer_id
GROUP BY o.customer_id
ORDER BY SUM(amount) DESC
LIMIT 2;

-- Difference between max and avg order amount
SELECT max(amount)-AVG(amount)
from orders;