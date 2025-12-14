# UGANDA MARTYRS UNIVERSITY
# DATA MINING AND BUSINESS INTELLIGENCE Practical Exam
# QUESTION 3: Association Rule Mining (Apriori Algorithm)
# Name: Baguma Gerald
# student number: 2023 - B072 - 31712

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# PART A: DATA PREPARATION

# Creating the dataset manually
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

# Converting to DataFrame
df = pd.DataFrame(data)

# Displaying the dataset
print("Original Transaction Dataset:")
print(df)

# Converting strings into lists
transactions = []
for items in df["Items"]:
    transaction = [item.strip() for item in items.split(",")]
    transactions.append(transaction)

print("\nTransactions:")
print(transactions)

# One-hot encode the dataset
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(encoded_array, columns=encoder.columns_)

print("\nOne-Hot Encoded Dataset:")
print(df_encoded)

# PART B: APRIORI ALGORITHM

# Applying Apriori with minimum support of 0.2
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generating association rules with confidence ≥ 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Selecting key columns
rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

print("\nAssociation Rules:")
print(rules)
