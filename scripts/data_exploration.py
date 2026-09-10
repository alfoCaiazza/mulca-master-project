import pandas as pd

x = 10
df = pd.read_csv("../data/filtered_wildchat.csv")
print(f"Number of total conversations: {len(df)}")
print(f"Number of conversations with more than {x} turns: {len(df[df['turn'] > x])}")

print("\n")
print("Example of a medium-lenght conversation:")
print(df[df['turn'] > x].iloc[0])