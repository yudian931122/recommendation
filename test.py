import numpy as np
import pandas as pd

df = pd.DataFrame([['yes'], ['no']], columns=['label'])
print(df)

print("*" * 20)
print(df['label'].apply(lambda x: "yes" in x).astype(int))
