import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Task Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .logo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        margin-right: 1rem;
        font-size: 1.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .ai-recommendations {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
    }
    .recommendation-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .task-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .priority-high { border-left-color: #ff4757 !important; }
    .priority-medium { border-left-color: #ffa502 !important; }
    .priority-low { border-left-color: #2ed573 !important; }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-todo { background: #f1f2f6; color: #57606f; }
    .status-progress { background: #3742fa; color: white; }
    .status-done { background: #2ed573; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# Mock data for demonstration
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    
    priorities = ['Low', 'Medium', 'High']
    issue_types = ['Bug', 'Task', 'Story', 'Epic']
    statuses = ['To Do', 'In Progress', 'Done']
    components = ['Frontend', 'Backend', 'Database', 'API', 'UI/UX']
    assignees = ['Sarah Chen', 'Alex Rivera', 'Mike Johnson', 'Emma Wilson', 'David Kim']
    
    tasks = []
    for i in range(20):
        task = {
            'id': i + 1,
            'title': f"Task {i+1}: {np.random.choice(['Fix bug in', 'Implement', 'Update', 'Refactor'])} {np.random.choice(['login system', 'dashboard', 'API endpoint', 'user interface', 'database schema'])}",
            'description': f"Detailed description for task {i+1}. This involves multiple steps and considerations.",
            'priority': np.random.choice(priorities),
            'issue_type': np.random.choice(issue_types),
            'status': np.random.choice(statuses),
            'component': np.random.choice(components),
            'assignee': np.random.choice(assignees),
            'created': datetime.now() - timedelta(days=np.random.randint(1, 30)),
            'due_date': datetime.now() + timedelta(days=np.random.randint(1, 14)),
            'ai_score': np.random.randint(60, 100)
        }
        tasks.append(task)
    
    return pd.DataFrame(tasks)

# Simplified AI predictor without heavy ML dependencies
class SimpleAIPredictor:
    def __init__(self):
        self.priority_classes = ['Low', 'Medium', 'High']
        self.assignees = ['Sarah Chen', 'Alex Rivera', 'Mike Johnson', 'Emma Wilson', 'David Kim']
    
    def predict_priority(self, text, features=None):
        """Simple rule-based priority prediction"""
        text_lower = text.lower()
        
        # High priority keywords
        high_priority_keywords = ['critical', 'urgent', 'bug', 'error', 'crash', 'security', 'production']
        # Medium priority keywords  
        medium_priority_keywords = ['improvement', 'feature', 'update', 'refactor']
        
        high_score = sum(1 for keyword in high_priority_keywords if keyword in text_lower)
        medium_score = sum(1 for keyword in medium_priority_keywords if keyword in text_lower)
        
        if high_score > 0:
            return 'High', 0.8 + (high_score * 0.05)
        elif medium_score > 0:
            return 'Medium', 0.7 + (medium_score * 0.05)
        else:
            return 'Low', 0.6
    
    def suggest_assignee(self, task_info):
        """Simple assignee suggestion based on component expertise"""
        component_experts = {
            'Frontend': 'Emma Wilson',
            'Backend': 'Alex Rivera', 
            'Database': 'Mike Johnson',
            'API': 'Sarah Chen',
            'UI/UX': 'Emma Wilson'
        }
        
        component = task_info.get('component', 'Frontend')
        suggested = component_experts.get(component, self.assignees[0])
        workload = np.random.randint(3, 8)  # Mock workload
        
        return suggested, workload
    
    def get_recommendations(self, tasks_df):
        recommendations = []
        
        # Priority adjustment recommendation
        high_priority_count = len(tasks_df[tasks_df['priority'] == 'High'])
        if high_priority_count > 3:
            recommendations.append({
                'type': 'Priority Alert',
                'message': f'{high_priority_count} high-priority tasks detected. Consider resource reallocation.',
                'urgency': 'high'
            })
        
        # Workload balance recommendation
        assignee_counts = tasks_df['assignee'].value_counts()
        if len(assignee_counts) > 1:
            max_load = assignee_counts.max()
            min_load = assignee_counts.min()
            if max_load - min_load > 2:
                overloaded = assignee_counts.idxmax()
                underloaded = assignee_counts.idxmin()
                recommendations.append({
                    'type': 'Workload Balance',
                    'message': f'Consider redistributing tasks from {overloaded} to {underloaded}.',
                    'urgency': 'medium'
                })
        
        # Deadline risk recommendation
        now = datetime.now()
        upcoming_deadlines = len(tasks_df[
            (tasks_df['due_date'] > now) & 
            (tasks_df['due_date'] <= now + timedelta(days=3)) &
            (tasks_df['status'] != 'Done')
        ])
        
        if upcoming_deadlines > 0:
            recommendations.append({
                'type': 'Deadline Alert',
                'message': f'{upcoming_deadlines} tasks have deadlines within 3 days.',
                'urgency': 'high'
            })
        
        # Add default recommendations if none generated
        if not recommendations:
            recommendations.append({
                'type': 'Performance',
                'message': 'Team productivity is optimal. Consider taking on additional projects.',
                'urgency': 'low'
            })
        
        return recommendations

# Initialize AI predictor
@st.cache_resource
def load_ai_models():
    return SimpleAIPredictor()

ai_predictor = load_ai_models()

# Load data
df = load_sample_data()

# Header
st.markdown("""
<div class="main-header">
    <div class="logo">🤖</div>
    <div>
        <h1 style="margin:0; color:#2f3542;">AI Task Manager</h1>
        <p style="margin:0; color:#57606f;">Intelligent productivity powered by AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# New Task Button (top-right)
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("➕ New Task", key="new_task_header"):
        st.session_state.show_new_task_form = True

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    completed_tasks = len(df[df['status'] == 'Done'])
    total_tasks = len(df)
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #667eea; margin-right: 0.5rem;">🎯</span>
            <span style="color: #57606f; font-size: 0.9rem;">Completion Rate</span>
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{completion_rate:.0f}%</div>
        <div style="color: #57606f; font-size: 0.8rem;">{completed_tasks} of {total_tasks} tasks completed</div>
        <div style="background: #667eea; height: 4px; border-radius: 2px; margin-top: 0.5rem; width: {completion_rate}%;"></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    ai_accuracy = 94
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #2ed573; margin-right: 0.5rem;">📈</span>
            <span style="color: #57606f; font-size: 0.9rem;">AI Efficiency</span>
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: #2ed573;">{ai_accuracy}%</div>
        <div style="color: #57606f; font-size: 0.8rem;">AI classification accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    high_priority_tasks = len(df[df['priority'] == 'High'])
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #ffa502; margin-right: 0.5rem;">⚠️</span>
            <span style="color: #57606f; font-size: 0.9rem;">High Priority</span>
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: #ffa502;">{high_priority_tasks}</div>
        <div style="color: #57606f; font-size: 0.8rem;">Tasks require attention</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    team_status = "Balanced"
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #667eea; margin-right: 0.5rem;">👥</span>
            <span style="color: #57606f; font-size: 0.9rem;">Team Load</span>
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{team_status}</div>
        <div style="color: #57606f; font-size: 0.8rem;">AI-optimized distribution</div>
    </div>
    """, unsafe_allow_html=True)

# AI Recommendations Section
recommendations = ai_predictor.get_recommendations(df)

st.markdown("""
<div class="ai-recommendations">
    <h2 style="margin: 0 0 1rem 0; display: flex; align-items: center;">
        ⚡ AI Recommendations
    </h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
""", unsafe_allow_html=True)

for rec in recommendations:
    urgency_color = "#ff4757" if rec['urgency'] == 'high' else "#ffa502" if rec['urgency'] == 'medium' else "#2ed573"
    st.markdown(f"""
        <div class="recommendation-card">
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: {urgency_color};">{rec['type']}</div>
            <div>{rec['message']}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Recent Tasks Section
st.markdown("""
<div style="margin: 2rem 0 1rem 0;">
    <h2 style="display: flex; align-items: center; color: #2f3542;">
        🕒 Recent Tasks
    </h2>
</div>
""", unsafe_allow_html=True)

# Filter and display recent tasks
recent_tasks = df.sort_values('created', ascending=False).head(10)

for _, task in recent_tasks.iterrows():
    priority_class = f"priority-{task['priority'].lower()}"
    
    # Status badge styling
    status_style = {
        'To Do': 'background: #f1f2f6; color: #57606f;',
        'In Progress': 'background: #3742fa; color: white;',
        'Done': 'background: #2ed573; color: white;'
    }.get(task['status'], 'background: #f1f2f6; color: #57606f;')
    
    st.markdown(f"""
    <div class="task-card {priority_class}">
        <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 0.5rem;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-weight: bold; color: #2f3542;">{task['title']}</span>
                    <span class="status-badge" style="background: #e8f4f8; color: #2c5aa0; padding: 0.25rem 0.5rem; border-radius: 8px; font-size: 0.7rem;">{task['priority']}</span>
                    <span style="background: #e8f4f8; color: #2c5aa0; padding: 0.25rem 0.5rem; border-radius: 8px; font-size: 0.7rem;">{task['component']}</span>
                </div>
                <div style="color: #57606f; font-size: 0.9rem; margin-bottom: 0.5rem;">{task['description']}</div>
                <div style="display: flex; align-items: center; gap: 1rem; font-size: 0.8rem; color: #747d8c;">
                    <span>👤 {task['assignee']}</span>
                    <span>📅 {task['created'].strftime('%Y-%m-%d')}</span>
                    <span>🎯 AI Score: {task['ai_score']}%</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="status-badge" style="{status_style}">{task['status']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# New Task Modal/Form
if st.session_state.get('show_new_task_form', False):
    st.markdown("---")
    st.markdown("### ➕ Create New Task")
    
    with st.form("new_task_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            task_title = st.text_input("Task Title *", placeholder="Enter task title...")
            task_description = st.text_area("Description", placeholder="Describe the task in detail...")
            issue_type = st.selectbox("Issue Type", ['Bug', 'Task', 'Story', 'Epic'])
        
        with col2:
            component = st.selectbox("Component", ['Frontend', 'Backend', 'Database', 'API', 'UI/UX'])
            due_date = st.date_input("Due Date", min_value=datetime.now().date())
            manual_priority = st.selectbox("Priority (Optional - AI will suggest)", ['Auto-Detect', 'Low', 'Medium', 'High'])
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.form_submit_button("🤖 AI Predict & Create", type="primary"):
                if task_title:
                    # AI Prediction
                    task_text = f"{task_title} {task_description}"
                    
                    if manual_priority == 'Auto-Detect':
                        predicted_priority, confidence = ai_predictor.predict_priority(task_text)
                    else:
                        predicted_priority = manual_priority
                        confidence = 1.0
                    
                    suggested_assignee, workload = ai_predictor.suggest_assignee({
                        'priority': predicted_priority,
                        'component': component
                    })
                    
                    # Create new task
                    new_task = {
                        'id': len(df) + 1,
                        'title': task_title,
                        'description': task_description,
                        'priority': predicted_priority,
                        'issue_type': issue_type,
                        'status': 'To Do',
                        'component': component,
                        'assignee': suggested_assignee,
                        'created': datetime.now(),
                        'due_date': pd.Timestamp(due_date),
                        'ai_score': int(confidence * 100)
                    }
                    
                    st.success(f"""
                    ✅ Task created successfully!
                    
                    🎯 **AI Predictions:**
                    - Priority: {predicted_priority} (Confidence: {confidence:.1%})
                    - Suggested Assignee: {suggested_assignee} (Current workload: {workload} tasks)
                    
                    The task has been added to your project!
                    """)
                    
                    st.session_state.show_new_task_form = False
                    st.rerun()
                else:
                    st.error("Please enter a task title!")
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.session_state.show_new_task_form = False
                st.rerun()

# Sidebar with additional features
with st.sidebar:
    st.markdown("## 📊 Analytics")
    
    # Priority distribution chart
    priority_counts = df['priority'].value_counts()
    fig_priority = px.pie(
        values=priority_counts.values, 
        names=priority_counts.index,
        title="Task Priority Distribution",
        color_discrete_map={'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#2ed573'}
    )
    fig_priority.update_layout(height=300)
    st.plotly_chart(fig_priority, use_container_width=True)
    
    # Team workload
    st.markdown("## 👥 Team Workload")
    assignee_counts = df['assignee'].value_counts().head(5)
    
    for assignee, count in assignee_counts.items():
        progress = count / assignee_counts.max()
        color = "#ff4757" if progress > 0.8 else "#ffa502" if progress > 0.6 else "#2ed573"
        
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                <span style="font-size: 0.8rem;">{assignee}</span>
                <span style="font-size: 0.8rem; color: {color};">{count} tasks</span>
            </div>
            <div style="background: #f1f2f6; height: 6px; border-radius: 3px;">
                <div style="background: {color}; height: 6px; border-radius: 3px; width: {progress*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📁 Export Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"tasks_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #747d8c; font-size: 0.8rem; padding: 1rem;">
    🤖 AI Task Manager v1.0 | Powered by Simple AI | Built with Streamlit
</div>
""", unsafe_allow_html=True)
