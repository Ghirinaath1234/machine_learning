import streamlit as st
import pandas as pd
from classi import train_test,log,decision,random,ada,gradient
from regssi import train_test,logr,decisionr,randomr,adar,gradientr


st.title('Supervised Machine learning Model for EDA dataset')

datatype=st.sidebar.radio('Select the predictive values',['Regression','Classification'])

if datatype=="Classification":
    uploaded_file = st.file_uploader("Choose a best EDA file")

    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        st.dataframe(df.head(5))
        inde_var=st.radio('Select the independent variable',[i for i in df.columns])
        x1,x2,y1,y2=train_test(df,inde_var)
        inarr=[]
        for i in x1.columns:
            input=st.number_input(f'Enter the {i}')
            inarr.append(input)
        model_choice=st.selectbox('Choose a models',('LogisticRegression','DecisionTree','RandomForest','Adaboosting','GradientBoosting'))
        action=st.button("Show the models", type="primary")
        if action: 
            if model_choice=='LogisticRegression':
                model,test_acc,train_acc,report=log(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The train accurancy is {train_acc}')
                st.write(f'The train accurancy is {test_acc}')
                report_df=pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            elif model_choice=='DecisionTree':
                model,test_acc,train_acc,report=decision(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The train accurancy is {train_acc}')
                st.write(f'The train accurancy is {test_acc}')
                report_df=pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            elif model_choice=='Adaboosting':
                model,test_acc,train_acc,report=ada(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The train accurancy is {train_acc}')
                st.write(f'The train accurancy is {test_acc}')
                report_df=pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            elif model_choice=='RandomForest':
                model,test_acc,train_acc,report=random(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The train accurancy is {train_acc}')
                st.write(f'The train accurancy is {test_acc}')
                report_df=pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            else:
                model,test_acc,train_acc,report=gradient(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The train accurancy is {train_acc}')
                st.write(f'The train accurancy is {test_acc}')
                report_df=pd.DataFrame(report).transpose()
                st.dataframe(report_df)






else:
    uploaded_file = st.file_uploader("Choose a best EDA file")

    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        st.dataframe(df.head(5))
        inde_var=st.radio('Select the independent variable',[i for i in df.columns])
        x1,x2,y1,y2=train_test(df,inde_var)
        inarr=[]
        for i in x1.columns:
            input=st.number_input(f'Enter the {i}')
            inarr.append(input)
        model_choice=st.selectbox('Choose a models',('LinearRegression','DecisionTree','RandomForest','Adaboosting','GradientBoosting'))
        action=st.button("Show the models", type="primary")
        if action: 
            if model_choice=='LinearRegression':
                model,report=logr(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The r2 score is {report}')
            elif model_choice=='DecisionTree':
                model,report=decisionr(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The r2 score is {report}')
            elif model_choice=='Adaboosting':
                model,report=adar(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The r2 score is {report}')
            elif model_choice=='RandomForest':
                model,report=randomr(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The r2 score is {report}')
            else:
                model,report=gradientr(x1,x2,y1,y2)
                output=model.predict([inarr])
                st.write(f'The predicted value is {output}')
                st.write(f'The r2 score is {report}')
            
            



                    



