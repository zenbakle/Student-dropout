# Predicting Student Dropouts in Higher Education

**Problem Statement:**

The goal of this project is to develop a machine learning model that can predict student dropouts in a higher education institution based on a comprehensive set of features. The project addresses a critical issue in the education sector, as identifying potential dropouts early can lead to interventions that improve student retention rates and overall academic success.


**Dataset Description:**

The dataset includes the following features:
- Marital status
- Application mode
- Application order
- Course
- Daytime/evening attendance
- Previous qualification
- Nationality
- Mother's qualification
- Father's qualification
- Mother's occupation
- Father's occupation
- Displaced status
- Educational special needs
- Debtor status
- Tuition fees up-to-date
- Gender
- Scholarship holder status
- Age at enrollment
- International student status
- Curricular units information for two semesters
- Regional economic indicators (unemployment rate, inflation rate, GDP)
- Target variable: Student status (enrolled, graduated, dropped out)

**Project Objective:**

The main objective of this project is to develop a predictive model that can determine whether a student is likely to drop out of their program, allowing higher education institutions to take proactive measures to prevent dropouts. This predictive model can serve as a valuable tool for academic advisors, administrators, and policy-makers.

**Methodology:**

The project follows a structured methodology:
1. Data collection and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature selection and engineering
4. Model building and training
5. Model evaluation and performance metrics
6. Interpretation of results and insights
7. Model optimization and deployment

**Key Challenges/constraints**
- Time constraint to try out other models
- Managing and preprocessing a diverse set of features, including both numerical and categorical data.

**References:**

Valentim Realinho, Jorge Machado, Luís Baptista, & Mónica V. Martins. (2021). Predict students' dropout and academic success (1.0) [Data set](https://doi.org/10.5281/zenodo.5777340)


## Instructions for Running the Project:

### Prerequisites

- [Pipenv](https://pipenv.pypa.io/en/latest/install/)
- [Docker](https://www.docker.com/get-started)

To provide clear instructions for running the project with the given files on GitHub, you can create a README file that includes the following steps:


1. **Clone the Repository:**

   ```bash
   git clone https://github.com/zenbakle/Student-dropout.git
   cd Student-dropout
   ```

2. **Set Up the Environment:**

   Ensure you have Pipenv and Docker installed on your system.

   - If you don't have Pipenv installed, you can install it using pip:

     ```bash
     pip install pipenv
     ```

3. **Install Dependencies:**

   Use Pipenv to set up the project environment and install the necessary dependencies.

   ```bash
   pipenv install
   ```

4. **Activate the Virtual Environment:**

   To activate the virtual environment, use the following command:

   ```bash
   pipenv shell
   ```

5. **Run the Training Script:**

   Train the machine learning model using the `train.py` script. Make sure to provide the required input data.

   ```bash
   python train.py
   ```

   This will train the model and save it in a suitable format.


6. **Dockerize the Application (Optional):**

   If you want to containerize the project using Docker, you can build and run a Docker container:

   - Build the Docker image:

     ```bash
     docker build -t midterm .
     ```

   - Run the Docker container:

     ```bash
     docker run -p 9696:9696 midterm
     ```


8. **Access the Application:**

   You can access the app through a web browser or API requests, If you have deployed the project using Docker or
    With the Flask application.

   ```bash
   student = {'marital_status': 1.0, 
   'application_mode': 17.0,
   'application_order': 1.0,
    'course': 13.0,
    'daytime/evening_attendance': 1.0,
    'previous_qualification': 16.0,
    'nationality': 1.0,
    "mother's_qualification": 1.0,
    "father's_qualification": 14.0,
    "mother's_occupation": 5.0,
    "father's_occupation": 6.0,
    'displaced': 0.0,
    'educational_special_needs': 0.0,
    'debtor': 0.0,
    'uptodate_fees': 1.0,
    'gender': 0.0,
    'scholarship_holder': 1.0,
    'enrollment_age': 22.0,
    'international': 0.0,
    '1stsem_enrolled': 7.0,
    '1stsem_evaluations': 10.0,
    '1stsem_approved': 6.0,
    '1stsem_grade': 13.3142857142857,
    '2ndsem_enrolled': 8.0,
    '2ndsem_evaluations': 10.0,
    '2ndsem_approved': 7.0,
    '2ndsem_grade': 12.875}

   url = "http://0.0.0.0:9696//predict"
   requests.post(url, json=student).json()
   ```