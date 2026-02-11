import pandas as pd

grades = pd.Series([85, None, 92, 45, None, 78, 55])

missing = grades.isnull()

filled_grades = grades.fillna(0)

# Apply boolean mask (scores > 60)
filtered = filled_grades[filled_grades > 60]

# Print results
print("Original Series:")
print(grades)

print("\nMissing Values (True means missing):")
print(missing)

print("\nFilled Series (Missing replaced with 0):")
print(filled_grades)

print("\nScores Greater Than 60:")
print(filtered)

