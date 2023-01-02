import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 


import math


class Solution:

    def __init__(self):
        self.df = pd.read_csv("customer_churn.csv")
        self.dataframe = pd.read_csv("customer_churn.csv")

        enc = LabelEncoder()

        for i in self.df.columns:
            if self.df.dtypes[i]=='O':
                self.df[i] = enc.fit_transform(self.df[i])
                #print(f'{i} : {self.df[i].unique()}')
        
        normal=StandardScaler()
        for i in self.df.columns:
            if self.df.dtypes[i]=='float64':
                self.df[i]=normal.fit_transform(self.df[i][:, np.newaxis])


    def common_eda(self):

        self.df.drop(self.df[self.df["TotalCharges"]==" "].index, axis= 0,inplace=True)

        self.df[['TotalCharges']] = self.df[['TotalCharges']].apply(pd.to_numeric)

        self.df.pop("customerID")


    def home(self):
        st.markdown("# Welcome to our Customer Churn Predictor!!!")
        st.sidebar.markdown("# Home🏠")

        st.subheader("Enter Your Details in the forms Page of our Website to predict if your customer is going to churn the subscription")

        #st.write("In this Session you will build a website that takes values from the user,stores the value in a database and uses the data to predict whether the user will be regular in going to college.")

        st.header("Our Objective:")
        st.write("To find the people who are vurnerable to stop subscription from the telecom industry.")
        st.write("To evaluate the measures to retain customers")

        st.success("👈Choose the Forms option in the sidebar to predict your result.")
        st.success("👈Choose the Display Stats option to visualize the data collected so far")


    def forms(self):
        st.markdown("# Form📑")
        st.sidebar.markdown("# Form📑")

        gender = st.selectbox("Choose your Gender:",["Male","Female"])

        senior_citizen = st.selectbox("Are you a senior citizen: ",["Yes","No"])

        senior_map = {"Yes":0,"No":1}

        senior_citizen = senior_map[senior_citizen]

        partner = st.selectbox("Do you have a Partner?:",["Yes","No"])

        dependent = st.selectbox("Are you dependent on parents/guardians?:",["Yes","No"])

        tenure = st.number_input("Enter your Tenure period",5)

        phone_service = st.selectbox("Have you accessed the phone service?:",["Yes","No"])

        mutiple_lines = st.selectbox("Do you have Multiple Phone Lines:",["Yes","No","No Phone"])

        internet = st.selectbox("What is the type of internet Service you use?:",["DSL","Fiber optics","No"])

        onlineservice = st.selectbox("Do you have a Online Service:",["Yes","No","No internet service"])

        onlinebackup = st.selectbox("Do you have a online backup:",["Yes","No","No internet service"])

        deviceprotection = st.selectbox("Do you have Device Protection",["Yes","No","No internet service"])

        techsupport = st.selectbox("Do you have Tech Support:",["Yes","No","No internet service"])

        streaming = st.selectbox("Do you have Streaming:",["Yes","No","No internet service"])

        streaming_movies = st.selectbox("Do you stream movies:",["Yes","No","No internet service"])

        contract = st.selectbox("Select Your Contract:",["Month-to-month","One year","Two year"])

        paperless = st.selectbox("Do you have paperless service:",["Yes","No","No internet service"])

        paymentmethod = st.selectbox("Do you have Payment Method:",["Electric","Mailed check","Bank transfer (automatic)","Credit card (automatic)",""])

        monthlycharges = st.number_input("Enter your monthly charges:",0.0)

        totalcharges = st.number_input("Enter your total charges:",0.0)

        user_data = [gender,senior_citizen,partner,dependent,tenure,phone_service,mutiple_lines,internet,onlineservice,onlinebackup,deviceprotection,techsupport,streaming,streaming_movies,contract,paperless,paymentmethod,monthlycharges,totalcharges]

        if st.button("Predict the value using Decision tree"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            model = tree.DecisionTreeClassifier()

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))
        
        if st.button("Predict the value using Logistic Regression"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            model = LogisticRegression()

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))

        if st.button("Predict the value using KNN"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            k = math.floor(math.sqrt(self.df.shape[0]))
            
            model = KNeighborsClassifier(n_neighbors = k)

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))
    
        if st.button("Predict the value using Random Forest"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            model = RandomForestClassifier(n_estimators=10,criterion="entropy")

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))
    
        if st.button("Predict the value using AdaBoostClassifier:"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            model = AdaBoostClassifier(learning_rate=1,n_estimators=100)

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))

        if st.button("Predict the value using Naive Bayes:"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            model = GaussianNB()

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))
        
        if st.button("Predict the value using SVM"):

            enc = LabelEncoder()
            
            new_df = pd.DataFrame([user_data])

            for i in new_df.columns:
                if new_df[i].dtype == "O":
                    new_df[i] = enc.fit_transform(new_df[i])
            
            model = LogisticRegression()

            x = self.df.iloc[:,:-1]
 
            y = self.df.iloc[:,-1]

            x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.8)

            model.fit(x_train,y_train)

            y_predict = model.predict(new_df[new_df.columns])

            y_pred =  model.predict(x_test)

            #st.write(accuracy_score(y_pred,y_test))

            if(y_predict[0] == 0):
                st.success("You will not churn out of service this year")
                st.balloons()
            
            else:
                st.error("You will churn out of service this year")
                st.snow()
            
            st.write("General Accuracy of this model: ",accuracy_score(y_pred,y_test))





    def data_display(self):
        st.markdown("# Display Stats📈")
        st.sidebar.markdown("# Display Stats📈")

        st.header("Dataframe:")

        st.dataframe(self.df)

        st.header("BoxPlots:")

        st.subheader("Boxplots for TotalCharges:")

        fig,axes = plt.subplots()

        axes = sns.boxplot(y=self.df["TotalCharges"])

        st.pyplot(fig)
    
    def eda(self):

        st.header("Initial EDA done:")

        st.code("""self.df.drop(df[df["TotalCharges"]==" "].index, axis= 0,inplace=True)\nself.df[['TotalCharges']] = self.df[['TotalCharges']].apply(pd.to_numeric)\nself.df.pop("customerID")""")

    def data_preprocessing(self):

        st.header("Data Preprocessing Done on the dataset")
        
        st.code("""enc = LabelEncoder()\nfor i in self.df.columns:
            if self.df.dtypes[i]=='O':
                self.df[i] = enc.fit_transform(self.df[i])
                #print(f'{i} : {self.df[i].unique()}')\nnormal=StandardScaler()\nfor i in self.df.columns:
            if self.df.dtypes[i]=='float64':
                self.df[i]=normal.fit_transform(self.df[i][:, np.newaxis])
        """)



if __name__=="__main__":

    s = Solution()

    page_names_to_funcs = {
        "Home🏠": s.home,
        "Forms📑": s.forms,
        #"Data Visualization📈": s.data_display,
        "EDA": s.eda,
        "Data Preprocessing": s.data_preprocessing
    }

    s.common_eda()

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()