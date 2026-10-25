"""
Mock Database Engine for Live SQLite Execution.
Initializes realistic in-memory SQLite databases for Spider benchmarks
using Python's standard library sqlite3 without external DLL dependencies.
"""

import sqlite3
from typing import Dict, Any, Tuple, Optional, List


def create_sample_databases() -> Dict[str, sqlite3.Connection]:
    """
    Creates pre-populated SQLite connections for prominent Spider databases.
    """
    connections: Dict[str, sqlite3.Connection] = {}

    # 1. CONCERT_SINGER Database
    cs_conn = sqlite3.connect(":memory:", check_same_thread=False)
    cs_cursor = cs_conn.cursor()
    cs_cursor.executescript("""
        CREATE TABLE stadium (
            Stadium_ID INTEGER PRIMARY KEY,
            Location TEXT,
            Name TEXT,
            Capacity INTEGER,
            Highest INTEGER,
            Lowest INTEGER,
            Average INTEGER
        );
        INSERT INTO stadium VALUES 
            (1, 'London', 'Wembley Arena', 90000, 89000, 75000, 82000),
            (2, 'Manchester', 'Old Trafford', 75000, 74000, 68000, 71000),
            (3, 'Paris', 'Stade de France', 81000, 80000, 72000, 76000),
            (4, 'Berlin', 'Olympiastadion', 74000, 73000, 65000, 69000);

        CREATE TABLE singer (
            Singer_ID INTEGER PRIMARY KEY,
            Name TEXT,
            Country TEXT,
            Song_Name TEXT,
            Song_release_year INTEGER,
            Age INTEGER,
            Is_male INTEGER
        );
        INSERT INTO singer VALUES
            (1, 'Adele Adkins', 'France', 'Rolling in the Deep', 2010, 34, 0),
            (2, 'Ed Sheeran', 'United Kingdom', 'Shape of You', 2017, 32, 1),
            (3, 'Dua Lipa', 'United Kingdom', 'Levitating', 2020, 27, 0),
            (4, 'Stromae', 'France', 'Papaoutai', 2013, 38, 1),
            (5, 'Billie Eilish', 'United States', 'Bad Guy', 2019, 21, 0),
            (6, 'The Weeknd', 'Canada', 'Blinding Lights', 2019, 33, 1);

        CREATE TABLE concert (
            concert_ID INTEGER PRIMARY KEY,
            concert_name TEXT,
            Theme TEXT,
            Stadium_ID INTEGER,
            Year INTEGER,
            FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
        );
        INSERT INTO concert VALUES
            (1, 'World Tour 2020', 'Pop Euphoria', 1, 2020),
            (2, 'Acoustic Evenings', 'Chill Folk', 2, 2021),
            (3, 'Summer Vibes Fest', 'Electronic', 3, 2022);

        CREATE TABLE singer_in_concert (
            concert_ID INTEGER,
            Singer_ID INTEGER,
            PRIMARY KEY (concert_ID, Singer_ID),
            FOREIGN KEY (concert_ID) REFERENCES concert(concert_ID),
            FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
        );
        INSERT INTO singer_in_concert VALUES
            (1, 1), (1, 2), (2, 2), (2, 3), (3, 4), (3, 6);
    """)
    cs_conn.commit()
    connections["concert_singer"] = cs_conn

    # 2. PETS_1 Database
    pet_conn = sqlite3.connect(":memory:", check_same_thread=False)
    pet_cursor = pet_conn.cursor()
    pet_cursor.executescript("""
        CREATE TABLE Student (
            StuID INTEGER PRIMARY KEY,
            LName TEXT,
            Fname TEXT,
            Age INTEGER,
            Sex TEXT,
            Major TEXT,
            Advisor INTEGER,
            city_code TEXT
        );
        INSERT INTO Student VALUES
            (101, 'Smith', 'Linda', 21, 'F', 'CS', 1, 'NYC'),
            (102, 'Johnson', 'Mark', 23, 'M', 'Math', 2, 'BOS'),
            (103, 'Williams', 'Sarah', 20, 'F', 'Biology', 1, 'CHI'),
            (104, 'Brown', 'David', 22, 'M', 'CS', 3, 'NYC');

        CREATE TABLE Pets (
            PetID INTEGER PRIMARY KEY,
            PetType TEXT,
            pet_age INTEGER,
            weight REAL
        );
        INSERT INTO Pets VALUES
            (1, 'Dog', 4, 15.2),
            (2, 'Cat', 3, 4.5),
            (3, 'Parrot', 7, 0.8),
            (4, 'Dog', 6, 22.0);

        CREATE TABLE Has_Pet (
            StuID INTEGER,
            PetID INTEGER,
            PRIMARY KEY (StuID, PetID),
            FOREIGN KEY (StuID) REFERENCES Student(StuID),
            FOREIGN KEY (PetID) REFERENCES Pets(PetID)
        );
        INSERT INTO Has_Pet VALUES
            (101, 1), (101, 2), (102, 3), (104, 4);
    """)
    pet_conn.commit()
    connections["pets_1"] = pet_conn

    # 3. CAR_1 Database
    car_conn = sqlite3.connect(":memory:", check_same_thread=False)
    car_cursor = car_conn.cursor()
    car_cursor.executescript("""
        CREATE TABLE continents (
            ContId INTEGER PRIMARY KEY,
            Continent TEXT
        );
        INSERT INTO continents VALUES (1, 'America'), (2, 'Europe'), (3, 'Asia');

        CREATE TABLE countries (
            CountryId INTEGER PRIMARY KEY,
            CountryName TEXT,
            Continent INTEGER
        );
        INSERT INTO countries VALUES (1, 'USA', 1), (2, 'Germany', 2), (3, 'Japan', 3);

        CREATE TABLE car_makers (
            Id INTEGER PRIMARY KEY,
            Maker TEXT,
            FullName TEXT,
            Country TEXT
        );
        INSERT INTO car_makers VALUES (1, 'Ford', 'Ford Motor Co', 'USA'), (2, 'BMW', 'Bayerische Motoren Werke', 'Germany'), (3, 'Toyota', 'Toyota Motor Corp', 'Japan');

        CREATE TABLE cars_data (
            Id INTEGER PRIMARY KEY,
            MPG REAL,
            Cylinders INTEGER,
            Edispl REAL,
            Horsepower INTEGER,
            Weight REAL,
            Accelerate REAL,
            Year INTEGER
        );
        INSERT INTO cars_data VALUES 
            (1, 18.0, 8, 307.0, 130, 3504, 12.0, 1970),
            (2, 24.0, 4, 113.0, 95, 2372, 15.0, 1970),
            (3, 27.0, 4, 97.0, 88, 2130, 14.5, 1971),
            (4, 14.0, 8, 455.0, 225, 4425, 10.0, 1970);
    """)
    car_conn.commit()
    connections["car_1"] = car_conn

    # 4. STUDENTS Database
    students_conn = sqlite3.connect(":memory:", check_same_thread=False)
    students_cursor = students_conn.cursor()
    students_cursor.executescript("""
        CREATE TABLE students (
            student_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            major TEXT
        );
        INSERT INTO students VALUES 
            (1, 'Alice Smith', 20, 'Computer Science'),
            (2, 'Bob Jones', 22, 'Mathematics'),
            (3, 'Charlie Brown', 21, 'Physics');

        CREATE TABLE courses (
            course_id INTEGER PRIMARY KEY,
            course_name TEXT,
            department TEXT,
            credits INTEGER
        );
        INSERT INTO courses VALUES
            (101, 'Algorithms', 'Computer Science', 4),
            (102, 'Calculus III', 'Mathematics', 4),
            (103, 'Quantum Physics', 'Physics', 3);

        CREATE TABLE enrollments (
            enrollment_id INTEGER PRIMARY KEY,
            student_id INTEGER,
            course_id INTEGER,
            grade TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (course_id) REFERENCES courses(course_id)
        );
        INSERT INTO enrollments VALUES
            (1, 1, 101, 'A'),
            (2, 2, 102, 'A-'),
            (3, 3, 103, 'B+');
    """)
    students_conn.commit()
    connections["students"] = students_conn

    # 5. HOTEL Database
    hotel_conn = sqlite3.connect(":memory:", check_same_thread=False)
    hotel_cursor = hotel_conn.cursor()
    hotel_cursor.executescript("""
        CREATE TABLE hotel (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            role TEXT,
            salary INTEGER
        );
        INSERT INTO hotel VALUES 
            (1, 'John Doe', 42, 'manager', 85000),
            (2, 'Jane Doe', 28, 'receptionist', 45000),
            (3, 'Sam Wilson', 35, 'chef', 60000),
            (4, 'Emily Watson', 24, 'concierge', 40000);

        CREATE TABLE rooms (
            room_no INTEGER PRIMARY KEY,
            room_type TEXT,
            price_per_night REAL,
            status TEXT
        );
        INSERT INTO rooms VALUES
            (101, 'Deluxe Suite', 250.0, 'Occupied'),
            (102, 'Standard King', 150.0, 'Available'),
            (103, 'Executive Suite', 320.0, 'Occupied');
    """)
    hotel_conn.commit()
    connections["hotel"] = hotel_conn

    # 6. ECOMMERCE Database
    ecom_conn = sqlite3.connect(":memory:", check_same_thread=False)
    ecom_cursor = ecom_conn.cursor()
    ecom_cursor.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            city TEXT
        );
        INSERT INTO customers VALUES
            (1, 'Sarah Connor', 'sarah@example.com', 'Los Angeles'),
            (2, 'John Connor', 'john@example.com', 'New York'),
            (3, 'Kyle Reese', 'kyle@example.com', 'Chicago');

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            price REAL,
            stock INTEGER
        );
        INSERT INTO products VALUES
            (1, 'MacBook Pro', 'Electronics', 1999.99, 25),
            (2, 'Wireless Headphones', 'Electronics', 199.99, 100),
            (3, 'Ergonomic Desk Chair', 'Furniture', 349.50, 15);

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        INSERT INTO orders VALUES
            (1, 1, 1, 1, '2024-01-15'),
            (2, 2, 2, 2, '2024-02-10'),
            (3, 3, 3, 1, '2024-03-01');
    """)
    ecom_conn.commit()
    connections["ecommerce"] = ecom_conn

    # 7. COMPANY Database
    comp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    comp_cursor = comp_conn.cursor()
    comp_cursor.executescript("""
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT,
            budget INTEGER,
            location TEXT
        );
        INSERT INTO departments VALUES
            (1, 'Engineering', 1200000, 'San Francisco'),
            (2, 'Marketing', 450000, 'New York'),
            (3, 'Finance', 600000, 'Chicago');

        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            name TEXT,
            dept_id INTEGER,
            salary INTEGER,
            hire_year INTEGER,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );
        INSERT INTO employees VALUES
            (101, 'Alex Rivera', 1, 135000, 2020),
            (102, 'Jessica Chen', 1, 145000, 2019),
            (103, 'David Kim', 2, 92000, 2021),
            (104, 'Maria Garcia', 3, 105000, 2018);
    """)
    comp_conn.commit()
    connections["company"] = comp_conn

    return connections


# Global singleton connection cache
DB_CONNECTIONS = create_sample_databases()


def execute_sql(db_id: str, query: str) -> Tuple[Optional[List[str]], Optional[List[List[Any]]], Optional[str]]:
    """
    Executes a SQL query on the specified database.
    Returns:
        (column_headers, rows_list, error_message)
    """
    conn = DB_CONNECTIONS.get(db_id)
    if not conn:
        return None, None, f"Database '{db_id}' is not loaded in SQLite demo environment."

    clean_q = query.strip()
    if clean_q.startswith("```sql"):
        clean_q = clean_q[6:]
    if clean_q.startswith("```"):
        clean_q = clean_q[3:]
    if clean_q.endswith("```"):
        clean_q = clean_q[:-3]
    clean_q = clean_q.strip().rstrip(";")

    try:
        cursor = conn.cursor()
        cursor.execute(clean_q)
        headers = [desc[0] for desc in cursor.description] if cursor.description else ["Result"]
        rows = cursor.fetchall()
        # Convert tuples to lists for easy JSON/display compatibility
        rows_list = [list(r) for r in rows]
        return headers, rows_list, None
    except Exception as e:
        return None, None, str(e)


def get_connection_schema(db_id: str) -> Dict[str, Any]:
    """Inspects all tables and columns directly from the SQLite connection."""
    conn = DB_CONNECTIONS.get(db_id)
    if not conn:
        return {}
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall() if r[0] != "sqlite_sequence"]
    table_cols = {}
    for t in tables:
        cursor.execute(f"PRAGMA table_info({t});")
        cols = [col[1] for col in cursor.fetchall()]
        table_cols[t] = cols
    return {"tables": table_cols, "foreign_keys": []}


def register_custom_sql_database(db_id: str, sql_script: str) -> Tuple[bool, str]:
    """Registers an in-memory SQLite database by executing a DDL/DML SQL script."""
    clean_id = db_id.strip().lower().replace(" ", "_")
    if not clean_id:
        return False, "Database name cannot be empty."
    try:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.executescript(sql_script)
        conn.commit()
        DB_CONNECTIONS[clean_id] = conn
        return True, clean_id
    except Exception as e:
        return False, str(e)


def register_custom_json_database(db_id: str, json_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Registers an in-memory SQLite database from a JSON structure:
      {
        "tables": { "students": ["id", "name", "age"] },
        "data": { "students": [[1, "Alice", 20], [2, "Bob", 22]] }
      }
    """
    clean_id = db_id.strip().lower().replace(" ", "_")
    if not clean_id:
        return False, "Database name cannot be empty."
    try:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = conn.cursor()
        tables = json_data.get("tables", {})
        data = json_data.get("data", {})
        
        for tbl_name, cols in tables.items():
            col_defs = ", ".join(f"{c} TEXT" for c in cols)
            cursor.execute(f"CREATE TABLE {tbl_name} ({col_defs});")
            
            rows = data.get(tbl_name, [])
            if rows:
                placeholders = ", ".join("?" for _ in cols)
                cursor.executemany(f"INSERT INTO {tbl_name} VALUES ({placeholders})", rows)
                
        conn.commit()
        DB_CONNECTIONS[clean_id] = conn
        return True, clean_id
    except Exception as e:
        return False, str(e)
