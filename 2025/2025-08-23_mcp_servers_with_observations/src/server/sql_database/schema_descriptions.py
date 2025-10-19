CUSTOMERS_TABLE_DESC = """This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes."""
CUSTOMERS_TABLE_INFO_DESC = """CREATE TABLE CUSTOMERS (
    CustomerID INTEGER,
    Name VARCHAR,
    Age INTEGER,
    Email VARCHAR,
    Country VARCHAR
)

/* Add descriptions to CUSTOMERS table and its columns */
COMMENT ON TABLE CUSTOMERS IS 'This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.';
COMMENT ON COLUMN CUSTOMERS.CustomerID IS 'Unique identifier for each customer.';
COMMENT ON COLUMN CUSTOMERS.Name IS 'Name of the customer.';
COMMENT ON COLUMN CUSTOMERS.Age IS 'Age of the customer.';
COMMENT ON COLUMN CUSTOMERS.Email IS 'Email address of the customer.';
COMMENT ON COLUMN CUSTOMERS.Country IS 'Country of residence of the customer.';

/*
3 rows from CUSTOMERS table:
CustomerID    Name      Age    Email               Country
1             Alice     25     alice@example.com   USA
2             Bob       34     bob@example.com     Canada
3             Charlie   28     charlie@example.com UK
*/
"""
