import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Radius of Earth in miles
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


excel_path = 'EAW 2025 Remote Employee Closest Market or Office Location.xlsm'

xls = pd.ExcelFile(excel_path)
print("Available Sheets:", xls.sheet_names)

employee_df = pd.read_excel(excel_path, sheet_name='Remote Roster')

market_offices = employee_df[['Nearest Market Office', 'Lat', 'Long']].drop_duplicates().dropna()
market_offices.columns = ['Office Name', 'Office Lat', 'Office Long']


missing_coords = employee_df[employee_df[['Lat', 'Long']].isnull().any(axis=1)]
print(f"Employees with missing Lat or Long ({len(missing_coords)} rows):")
print(missing_coords[['Lat', 'Long']])


def find_closest_office(emp_lat, emp_long):
    if pd.isna(emp_lat) or pd.isna(emp_long):
        return pd.Series({'Closest Office': None, 'Distance to Closest Office (miles)': np.nan})
    distances = market_offices.apply(
        lambda row: haversine(emp_lat, emp_long, row['Office Lat'], row['Office Long']),
        axis=1
    )
    min_index = distances.idxmin()
    return pd.Series({
        'Closest Office': market_offices.loc[min_index, 'Office Name'],
        'Distance to Closest Office (miles)': distances[min_index]
    })

employee_df[['Closest Office', 'Distance to Closest Office (miles)']] = employee_df.apply(
    lambda row: find_closest_office(row['Lat'], row['Long']),
    axis=1
)


avg_by_office = employee_df.groupby('Closest Office')['Distance to Closest Office (miles)'].agg(['mean', 'count']).reset_index()
avg_by_office.columns = ['Office', 'Average Distance (miles)', 'Employee Count']

avg_by_manager = employee_df.groupby(['Manager', 'Closest Office'])['Distance to Closest Office (miles)'].mean().reset_index()
avg_by_manager.columns = ['Manager', 'Office', 'Avg Distance (miles)']

avg_by_cost = employee_df.groupby(['Cost Center', 'Closest Office'])['Distance to Closest Office (miles)'].mean().reset_index()
avg_by_cost.columns = ['Cost Center', 'Office', 'Avg Distance (miles)']


output_excel_path = 'combined_employee_distance_report.xlsx'

with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    employee_df.to_excel(writer, sheet_name='Employee Distances', index=False)
    avg_by_office.to_excel(writer, sheet_name='Avg Distance by Office', index=False)
    avg_by_manager.to_excel(writer, sheet_name='Avg Distance by Manager', index=False)
    avg_by_cost.to_excel(writer, sheet_name='Avg Distance by Cost Center', index=False)

print(f"Completed. Combined report saved as: {output_excel_path}")