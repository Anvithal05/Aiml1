import pandas as pd

products = pd.Series(
    [700, 150, 300],
    index=['Laptop', 'Mouse', 'Keyboard']
)

laptop_price = products['Laptop']

first_two = products.iloc[0:2]

print("Full Series:")
print(products)

print("\nPrice of Laptop:")
print(laptop_price)

print("\nFirst Two Products:")
print(first_two)
