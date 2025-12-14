# UGANDA MARTYRS UNIVERSITY
# DATA MINING AND BUSINESS INTELLIGENCE
# QUESTION 3: Association Rule Mining -Apriori Algorithm
# Name: Baguma Gerald

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# PART A: DATA PREPARATION

# Creating the dataset manually since it's small
data = {
    "TID": [1,2,3,4,5,6,7,8,9,10],
    "Items": [
        "Bread, Milk, Eggs",
        "Bread, Butter",
        "Milk, Diapers, Beer",
        "Bread, Milk, Butter",
        "Milk, Diapers, Bread",
        "Beer, Diapers",
        "Bread, Milk, Eggs, Butter",
        "Eggs, Milk",
        "Bread, Diapers, Beer",
        "Milk, Butter"
    ]
}
df = pd.DataFrame(data) #coverting the data into a structured table

# Converting the comma-separated strings into lists
transactions = []
for x in df["Items"]:
    items = [item.strip() for item in x.split(",")]
    transactions.append(items)

print("Transactions:\n", transactions[:5])

# One-hot encode the dataset for Apriori
encoder = TransactionEncoder()
encoded_arr = encoder.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(encoded_arr, columns=encoder.columns_)

print("\nOne-Hot Encoded Format:")
print(df_encoded.head())

# PART B: APRIORI ALGORITHM

# Running the Apriori algorithm with minimum support of 0.2
frequent_items = apriori(df_encoded, min_support=0.2, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_items)

# Generating association rules with confidence >= 0.5
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)

# Selecting only the key metrics
rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
print("\nAssociation Rules:")
print(rules)

# PART C: INTERPRETATION (TOP 3 RULES)

# Sorting rules by Lift to get the strongest ones
top_rules = rules.sort_values("lift", ascending=False).head(3)
print("\nTop 3 Strongest Rules Based on Lift:")

print(top_rules)

