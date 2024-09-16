from datasets import load_dataset, Dataset
import pandas as pd


dataset = load_dataset('csv', data_files="dataset/my_file.csv")
print(dataset)

my_dict = {"a": [1,2]}
dataset = Dataset.from_dict(my_dict)
print(dataset)


df = pd.DataFrame({"a": [1,2]})
dataset = Dataset.from_pandas(df)
print(dataset)

