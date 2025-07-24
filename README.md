This is a simple and interactive grading system built using Streamlit. It allows you to upload a CSV file containing student scores and choose between two grading methods: absolute grading (based on fixed thresholds like A = 90 and above) and relative grading (based on the normal distribution and percentiles). The app is helpful for educators, teaching assistants, or students who want to analyze class performance and visualize grading trends.

Features:
Upload a CSV file with StudentID and Score columns.

Choose between absolute or relative grading methods.

Visualize the score distribution using histograms, KDE plots, and normal distribution overlays.

View grade distributions and compare average scores by grade.

Check for outliers using IQR-based boxplots.

Download the final results (with assigned grades) as a CSV file.

Example Input Format:
The uploaded CSV should have the following format:

StudentID,Score
S001,85
S002,73
S003,92
S004,60
S005,48

Column names must be exactly: StudentID and Score.

How to Run:
Make sure you have Python installed.

Install the required libraries using pip:

pip install streamlit pandas numpy seaborn matplotlib scipy

Run the app using:

streamlit run grading_app.py

Then open the link shown in your terminal (usually http://localhost:8501) in your browser.

Grading Logic:
Absolute Grading: Assigns grades based on fixed thresholds.
A: 90+, B: 80+, C: 70+, D: 60+, F: below 60.

Relative Grading: Uses z-scores and percentiles to assign grades.
A: top 20%, B: next 30%, C: next 30%, D: next 10%, F: bottom 10%.

File Structure:
grading_app.py — the main Streamlit script

assets/ — optional folder for screenshots (not required)

Notes:
Works best with class-size datasets (e.g., 20–300 students).

You can extend it by adding weightages, customizable thresholds, or support for multiple assessments.

This app was created by me as a way to make grading analysis easier and more interactive.

If you'd like to suggest improvements or contribute, feel free to open a pull request or drop me a message. ill consider working more on it later in the future hopefully ig so people can use it for meaningful work and actual grading systems hopefully.
