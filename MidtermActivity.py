import streamlit as st
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import pip
pip.main(['install','seaborn'])

#outlier detection algorithm
def detect_outliers(data, threshold = 3):
    z_scores = numpy.abs((data-data.mean()) / data.std())
    return z_scores > threshold


#loading dataset
df = pandas.read_csv('dataset.csv')

#dropping unnecessary columns
df = df.drop(columns=['Date','Location'])

#duplicating a copy of original dataset
df_cleaned = df.copy()

for column in [col for col in df.columns if col != 'Rain Tomorrow']:
    outliers = detect_outliers(df[column])
    df_cleaned.loc[outliers,column] = numpy.nan


st.title("Midterm Activity")
st.caption("Macabecha, Jandel M.")
st.divider()
st.write("This is a dataset I got from kaggle titled 'USA Rain Prediction Dataset 2024 - 2025'. This dataset piqued my curiousity since I expect rain based mostly by cloud cover and wind speed while this dataset accounts for all known variables affecting the probability of rain. Through this data exploration, I hope to unveil wether cloud cover and wind speed accounts highly with regards to rain possibility.")
st.divider()
chart_data = pandas.DataFrame(
    df.iloc[:100,:]
)
st.subheader("Descriptive Stats")
st.write(df.describe())
st.divider()
st.subheader("Identified outliers via z-score")
st.write(df_cleaned.isna().sum())

st.divider()

sidebar = st.sidebar

with sidebar:
    var_radio = st.radio('Select rainfall predictor variable',df.columns[:-1])

fig, ax = plt.subplots()
seaborn.boxplot(data=df_cleaned[var_radio], ax=ax)
st.subheader(f"{var_radio} boxplot")
st.pyplot(fig)
left_col, right_col = st.columns(2)

with left_col:
    st.line_chart(chart_data,x=var_radio,y='Rain Tomorrow')
    
with right_col:
    st.scatter_chart(chart_data,x=var_radio,y='Rain Tomorrow')

st.divider()
st.subheader("Initial regression to identify p-values using MS Excel")
st.image("p-val.jpg")
st.caption("Variables with p-values greater than 0.005 indicates that tehre is insufficient evidence to conclude that the variable has a statistically signifcat relationship with the dependent variable which is Rain Tomorrow.")

st.divider()
st.subheader("Pairplot")
pplot = seaborn.pairplot(df[:200],hue='Rain Tomorrow', kind="reg", diag_kind="kde", plot_kws={'ci':None})
st.pyplot(pplot)
