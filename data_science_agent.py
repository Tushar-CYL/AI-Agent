import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

class DataScienceAgent:
    
    def __init__(self, data=None):
        self.data = data

    def load_data(self, filepath):
        """Load the dataset from a given file path."""
        self.data = pd.read_csv(filepath)
        print("Data Loaded Successfully")
    
    def analyze_data(self):
        """Perform basic analysis of the data."""
        print("Data Info:")
        print(self.data.info())
        print("Data Summary Statistics:")
        print(self.data.describe())
        
        # Visualizing correlation matrix
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()
        
        return self.data

    def clean_data(self):
        """Handle missing values and encode categorical variables."""
        # Handling missing values
        imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        
        # Encoding categorical variables (simple example for one column)
        if 'Category' in self.data.columns:
            self.data['Category'] = self.data['Category'].map({'A': 1, 'B': 2, 'C': 3})
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_cols] = scaler.fit_transform(self.data[numerical_cols])
        
        print("Data Cleaned Successfully")
        return self.data

    def build_model(self, target_column):
        """Build and train a RandomForestClassifier model."""
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]

        # Splitting data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initializing and training the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Making predictions and evaluating the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.show()

        return model

    def debug_code(self, code):
        """A basic debugging method that checks for common errors."""
        try:
            exec(code)
            return "Code executed successfully."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def execute_task(self, task_type, *args, **kwargs):
        """Execute a specific task."""
        if task_type == 'analyze':
            return self.analyze_data()
        elif task_type == 'clean':
            return self.clean_data()
        elif task_type == 'model':
            return self.build_model(*args, **kwargs)
        elif task_type == 'debug':
            return self.debug_code(*args, **kwargs)
        else:
            return "Invalid task type."

# Example Usage

# Initialize the agent
agent = DataScienceAgent()

# Load data
agent.load_data('Credit_Card_Applications.csv')

# Analyze data
agent.execute_task('analyze')

# Clean data
agent.execute_task('clean')

# Build model (assuming 'Target' is the target column)
agent.execute_task('model', target_column='Class')

# Debugging an example code
code_to_debug = """
import pandas as pd
data = pd.read_csv('non_existent_file.csv')
"""
print(agent.execute_task('debug', code_to_debug))
