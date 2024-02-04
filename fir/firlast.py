import pandas as pd

df = pd.read_csv("firfinal26")
df = df.fillna("")

text_col = []
for _, row in df.iterrows():
    prompt = "Below is an instruction that describes a task, paired with"
    Description = row["New Description"]
    section = row["section"]

    if len(Description.strip()) == 0:
        text = prompt + "### Description \n" + Description + "\n###section" + section
    else:
        break

    text_col.append(text)

df.loc[:, "text"] = text_col
print(df.head())

df.to_csv("train.csv", index=False)