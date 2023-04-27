import streamlit as st
import pandas as pd

df = pd.read_csv('laptop.csv')
df = df[df['Processor'] != 'Apple M1 Max']
df = df[df['Processor'] != 'Qualcomm Snapdragon']
df = df[df['Processor'] != 'AMD']
l = df['Brand'].unique()


brand = st.selectbox(
    'Laptop Brands',(l))
st.write('You selected:', brand)

df1 =  df[df['Brand']=='APPLE']
df2 = df[df['Brand']!='APPLE']



if brand == 'APPLE':

    ra = df1['RAM'].unique()
    ram = st.selectbox('RAM',(ra))
    st.write('You selected:', ram)
    
    rt = df1['RAM_type'].unique()
    ramt = st.selectbox('RAM_type',(rt))
    st.write('You selected:', ramt)

    pc = df1['Processor'].unique()
    processor = st.selectbox(
        'Processor',(pc))
    st.write('You selected:', processor)

    c = df1['Core'].unique()
    core = st.selectbox(
        'Core',(c))
    st.write('You selected:', core)

    Storage = st.number_input('Storage')
    st.write('Storage ', Storage)

    stby = st.selectbox('Storage bytes',('GB','TB'))
    st.write('You selected:', stby)

    stt = df1['Storagetype'].unique()
    stty = st.selectbox(
        'Storage Type',(stt))
    st.write('You selected:', stty)

    os = df1['OS'].unique()
    OS = st.selectbox(
        'OS',(os))
    st.write('You selected:', OS)

    ss = df1['Screen_Size'].unique()
    SS = st.selectbox(
        'Screen Size',(ss))
    st.write('You selected:', SS)
else:
    ra = df2['RAM'].unique()
    ram = st.selectbox('RAM',(ra))
    st.write('You selected:', ram)

    rt = df2['RAM_type'].unique()
    ramt = st.selectbox('RAM_type',(rt))
    st.write('You selected:', ramt)

    pc = df2['Processor'].unique()
    processor = st.selectbox('Processor',(pc))
    st.write('You selected:', processor)

    c = df2['Core'].unique()
    core = st.selectbox('Core',(c))
    st.write('You selected:', core)

    Storage = st.number_input('Storage')
    st.write('Storage ', Storage)

    stby = st.selectbox('Storage bytes',('GB','TB'))
    st.write('You selected:', stby)

    stt = df2['Storagetype'].unique()
    stty = st.selectbox(
        'Storage Type',(stt))
    st.write('You selected:', stty)

    os = df2['OS'].unique()
    OS = st.selectbox(
        'OS',(os))
    st.write('You selected:', OS)

    ss = df2['Screen_Size'].unique()
    SS = st.selectbox(
        'Screen Size',(ss))
    st.write('You selected:', SS)


if stby == 'TB':
    Storage = int(Storage)*1000
else:
    Storage = int(Storage)



###############################################################################################################

# X-Y train & test

y = df['MRP']
X = df[['Brand','Processor', 'Core', 'RAM','RAM_type', 'Storage(GB)', 'Storagetype', 'OS', 'Screen_Size']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=50,stratify=df['Processor'])

X_train_cat = X_train.select_dtypes(include=['object'])
X_train_num = X_train.select_dtypes(include=['int64', 'float64'])

#################################################
#Data PreProcessing on Train data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_num_rescaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns = X_train_num.columns, index = X_train_num.index)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_cat_ohe = pd.DataFrame(encoder.fit_transform(X_train_cat), columns=encoder.get_feature_names_out(X_train_cat.columns), index = X_train_cat.index)

X_train_transformed = pd.concat([X_train_num_rescaled, X_train_cat_ohe], axis=1)

###################################################
#Data PreProcessing on Test data

X_test.loc[800] = {'Brand':brand,"Processor" : processor, 'Core' : core, 'RAM' : ram, 'RAM_type' : ramt, 'Storage(GB)' : Storage , "Storagetype" : stty,'OS' : OS,'Screen_Size' : SS }
y_test.loc[800] = 0

X_test_cat = X_test.select_dtypes(include=['object'])
X_test_num = X_test.select_dtypes(include=['int64', 'float64'])

X_test_num_rescaled = pd.DataFrame(scaler.transform(X_test_num), columns = X_test_num.columns, index = X_test_num.index)

X_test_cat_ohe = pd.DataFrame(encoder.transform(X_test_cat), columns=encoder.get_feature_names_out(X_test_cat.columns), index = X_test_cat.index)

X_test_transformed = pd.concat([X_test_num_rescaled, X_test_cat_ohe], axis=1)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train_transformed, y_train)

y_test_pred = regressor.predict(X_test_transformed)
temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
result = int(temp_df["Predicted"][800])
if st.button('Submit'):
    st.header(f"Laptop Price is {result}")

