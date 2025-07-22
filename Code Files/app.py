import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from task_management_system import TaskManagementSystem

# Initialize session state
if 'task_system' not in st.session_state:
    st.session_state.task_system = TaskManagementSystem()
    st.session_state.task_system.load_data()
    st.session_state.task_system.train_models()

def create_workload_chart(workload_data):
    fig = px.bar(
        x=workload_data.index,
        y=workload_data.values,
        title='Current Workload Distribution',
        labels={'x': 'Assignee', 'y': 'Number of Active Tasks'}
    )
    return fig

def create_priority_chart(df):
    priority_counts = df['Priority'].value_counts()
    fig = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title='Task Priority Distribution'
    )
    return fig

def main():
    st.title('AI Task Management System')
    
    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select Page', ['Task Prediction', 'System Analytics', 'Workload Overview'])
    
    if page == 'Task Prediction':
        st.header('Task Prediction and Assignment')
        
        # Input form
        with st.form('task_form'):
            task_summary = st.text_area('Task Summary', 'Enter task description here...')
            col1, col2 = st.columns(2)
            
            with col1:
                issue_type = st.selectbox(
                    'Issue Type',
                    st.session_state.task_system.encoders['Issue Type'].classes_
                )
                status = st.selectbox(
                    'Status',
                    st.session_state.task_system.encoders['Status'].classes_
                )
            
            with col2:
                component = st.selectbox(
                    'Component',
                    st.session_state.task_system.encoders['Component'].classes_
                )
                duration = st.number_input('Duration (days)', min_value=1, max_value=30, value=7)
            
            submit_button = st.form_submit_button('Predict')
            
            if submit_button:
                # Prepare input features (mock text features for demo)
                # In production, you'd need to implement text feature extraction
                text_features = np.random.rand(768)  # BERT feature size
                
                additional_info = {
                    'Issue Type': issue_type,
                    'Status': status,
                    'Component': component,
                    'Duration': duration
                }
                
                # Make prediction
                prediction = st.session_state.task_system.predict_task(
                    text_features,
                    additional_info
                )
                
                # Display results
                st.success('Task Analysis Complete!')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Predicted Priority', prediction['priority'])
                    st.metric('Priority Confidence', f"{prediction['priority_confidence']:.2%}")
                
                with col2:
                    st.metric('Predicted Type', prediction['issue_type'])
                    st.metric('Type Confidence', f"{prediction['type_confidence']:.2%}")
                
                st.subheader('Assignment Suggestion')
                st.metric('Suggested Assignee', prediction['suggested_assignee'])
                
                # Assignment details
                details = prediction['assignment_details']
                st.write('Assignment Details:')
                st.write(f"- Current Workload: {details['current_tasks']} tasks")
                st.write(f"- Workload Score: {details['workload_score']:.2f}")
                st.write(f"- Task Complexity: {details['task_complexity']:.2f}")
    
    elif page == 'System Analytics':
        st.header('System Analytics')
        
        # Get data
        df = st.session_state.task_system.df
        
        # Priority distribution
        st.subheader('Task Priority Distribution')
        priority_fig = create_priority_chart(df)
        st.plotly_chart(priority_fig)
        
        # Issue type distribution
        issue_counts = df['Issue Type'].value_counts()
        st.subheader('Issue Type Distribution')
        issue_fig = px.bar(
            x=issue_counts.index,
            y=issue_counts.values,
            title='Issue Type Distribution'
        )
        st.plotly_chart(issue_fig)
        
        # Component analysis
        st.subheader('Component Analysis')
        component_counts = df['Component'].value_counts().head(10)
        component_fig = px.bar(
            x=component_counts.index,
            y=component_counts.values,
            title='Top 10 Components'
        )
        st.plotly_chart(component_fig)
        
        # Duration analysis
        st.subheader('Task Duration Analysis')
        duration_fig = px.box(
            df,
            x='Priority',
            y='Duration',
            title='Task Duration by Priority'
        )
        st.plotly_chart(duration_fig)
    
    else:  # Workload Overview
        st.header('Workload Overview')
        
        # Current workload
        workload = st.session_state.task_system.workload_manager.user_workload
        st.subheader('Current Workload Distribution')
        workload_fig = create_workload_chart(workload)
        st.plotly_chart(workload_fig)
        
        # Workload metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Active Tasks', workload.sum())
        with col2:
            st.metric('Average Tasks per Person', f"{workload.mean():.1f}")
        with col3:
            st.metric('Max Tasks per Person', workload.max())
        
        # Workload table
        st.subheader('Detailed Workload')
        workload_df = pd.DataFrame({
            'Assignee': workload.index,
            'Active Tasks': workload.values,
            'Workload %': (workload / workload.sum() * 100).round(1)
        })
        st.dataframe(workload_df)

if __name__ == '__main__':
    main() 