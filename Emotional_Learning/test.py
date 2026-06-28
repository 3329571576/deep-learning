import pandas as pd

df = pd.read_excel("taptap_review_ready.xlsx", sheet_name="Sheet1")

# 提取末尾数字（假定是最后一个字符）
df['sentiment'] = df['review'].str[-1]

# 去掉末尾的逗号和数字（最后两个字符）
df['review'] = df['review'].str[:-2]

# 保存
df.to_excel("整理后.xlsx", index=False)