import pandas as pd
import matplotlib.pyplot as plt

print('file name: ', __file__)
# creating a data frame
df = pd.read_csv("sample_count_per_class_pleural_effusion.csv")
print(df.head())

import seaborn as sns

sns.set(font_scale=1.2)
sns.set_palette("colorblind")

plt.figure()
plt.title('Sample counts per class for Pleural Effusion')
ax = sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
plt.xticks(ha='center')
ax.set_xticklabels(
    ('unmentioned\n0', 'positive\n1', 'uncertain\n2', 'negative\n3'),
    rotation=0, fontsize="10", ha="center")
# set the labels with values on top of the columns
for index, row in df.iterrows():
    print(index, type(row), row)
    print('row.name: ', row.name)
    print('row.values: ', row.values)
    ax.text(
        row.name,
        row.values[1],
        row.values[1],
        color='black',
        ha="center")

plt.savefig(
    'sample_count_per_class_pleural_effusion.pdf',
    bbox_inches='tight',
    pad_inches=0.05)
