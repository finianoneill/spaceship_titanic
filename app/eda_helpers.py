"""
Helper functions to be performed during EDA.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import LabelEncoder

# Function to load and perform initial data examination
def load_and_examine_data(df):
    """Load dataset and perform initial examination"""
    # df = pd.read_csv(filepath)
    
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nSummary Statistics:")
    print(df.describe(include='all').T)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 
                                 'Percentage': missing_percentage})
    print("\nMissing Values:")
    print(missing_data[missing_data['Missing Values'] > 0].sort_values(
        by='Missing Values', ascending=False))
    
    return df

# Function to visualize missing data
def visualize_missing_data(df):
    """Create visualizations for missing data"""
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Missing Value Matrix', fontsize=16)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    msno.heatmap(df)
    plt.title('Missing Value Correlation Heatmap', fontsize=16)
    plt.show()

# Function to extract features from complex columns
def feature_engineering(df):
    """Extract additional features from existing columns"""
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Extract deck, cabin number and side from Cabin column
    if 'Cabin' in df_processed.columns:
        df_processed['Cabin_missing'] = df_processed['Cabin'].isnull().astype(int)
        
        # Only process where Cabin is not null
        cabin_data = df_processed[~df_processed['Cabin'].isnull()]['Cabin'].str.split('/', expand=True)
        if cabin_data.shape[1] >= 3:  # Ensure we have all expected components
            df_processed.loc[~df_processed['Cabin'].isnull(), 'Deck'] = cabin_data[0]
            df_processed.loc[~df_processed['Cabin'].isnull(), 'Cabin_num'] = cabin_data[1].astype(int)
            df_processed.loc[~df_processed['Cabin'].isnull(), 'Side'] = cabin_data[2]
    
    # Extract group size and group id from PassengerId
    if 'PassengerId' in df_processed.columns:
        df_processed['GroupId'] = df_processed['PassengerId'].str.split('_', expand=True)[0]
        df_processed['GroupSize'] = df_processed.groupby('GroupId')['PassengerId'].transform('count')
        df_processed['PersonalId'] = df_processed['PassengerId'].str.split('_', expand=True)[1].astype(int)
    
    # Extract first and last name
    if 'Name' in df_processed.columns:
        names = df_processed['Name'].str.split(' ', n=1, expand=True)
        df_processed['FirstName'] = names[0]
        df_processed['LastName'] = names[1]
    
    # Total spending across all amenities
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df_processed['TotalSpending'] = df_processed[spending_cols].sum(axis=1)
    df_processed['NumAmenitiesUsed'] = (df_processed[spending_cols] > 0).sum(axis=1)
    
    return df_processed

# Function to analyze categorical variables
def analyze_categorical_vars(df, target_var='Transported'):
    """Analyze categorical variables and their relationship with target"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_var in categorical_cols:
        categorical_cols.remove(target_var)
    
    for col in categorical_cols:
        if df[col].nunique() < 30:  # Only plot if fewer than 30 unique values
            plt.figure(figsize=(12, 6))
            
            # Count plot
            plt.subplot(1, 2, 1)
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xticks(rotation=45)
            
            # Target relationship
            plt.subplot(1, 2, 2)
            sns.countplot(x=col, hue=target_var, data=df)
            plt.title(f'{col} vs {target_var}', fontsize=14)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Calculate proportions
            prop_df = pd.crosstab(df[col], df[target_var], normalize='index')
            plt.figure(figsize=(10, 5))
            prop_df.plot(kind='bar', stacked=True)
            plt.title(f'Proportion of {target_var} by {col}', fontsize=14)
            plt.ylabel('Proportion')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Function to analyze numerical variables
def analyze_numerical_vars(df, target_var='Transported'):
    """Analyze numerical variables and their relationship with target"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_var in numerical_cols:
        numerical_cols.remove(target_var)
    
    # Distribution plots for each numerical variable
    for col in numerical_cols:
        plt.figure(figsize=(15, 6))
        
        # Distribution
        plt.subplot(1, 3, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}', fontsize=14)
        
        # Box plot by target
        plt.subplot(1, 3, 2)
        sns.boxplot(x=target_var, y=col, data=df)
        plt.title(f'{col} by {target_var}', fontsize=14)
        
        # Violin plot by target
        plt.subplot(1, 3, 3)
        sns.violinplot(x=target_var, y=col, data=df)
        plt.title(f'Violin plot of {col} by {target_var}', fontsize=14)
        
        plt.tight_layout()
        plt.show()

# Function to analyze correlations
def analyze_correlations(df):
    """Analyze correlations between numerical variables"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlations
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Function for pairplot of important features
def plot_pairplot(df, target_var='Transported', features=None):
    """Create pairplot for selected features"""
    if features is None:
        # Select a subset of numerical features if none specified
        features = df.select_dtypes(include=['int64', 'float64']).columns[:5].tolist()
    
    if target_var not in features:
        features.append(target_var)
    
    plt.figure(figsize=(15, 15))
    sns.pairplot(df[features], hue=target_var, diag_kind='kde', 
                 plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5})
    plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to analyze target variable
def analyze_target(df, target_var='Transported'):
    """Analyze target variable distribution"""
    plt.figure(figsize=(12, 5))
    
    # Distribution of target variable
    plt.subplot(1, 2, 1)
    target_counts = df[target_var].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title(f'Distribution of {target_var}', fontsize=14)
    plt.xlabel(target_var)
    plt.ylabel('Count')
    
    # Percentage distribution
    plt.subplot(1, 2, 2)
    target_percentages = (target_counts / len(df)) * 100
    sns.barplot(x=target_percentages.index, y=target_percentages.values)
    plt.title(f'Percentage Distribution of {target_var}', fontsize=14)
    plt.xlabel(target_var)
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plt.show()

# Function to analyze bivariate relationships
def analyze_bivariate_relationships(df, target_var='Transported'):
    """Analyze relationships between pairs of variables"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_var in numerical_cols:
        numerical_cols.remove(target_var)
    
    # Select top spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
    spending_cols = [col for col in spending_cols if col in df.columns]
    
    # Create scatterplot matrix for spending features
    if len(spending_cols) > 1:
        g = sns.PairGrid(df.sample(min(1000, len(df))), 
                         vars=spending_cols, hue=target_var, 
                         diag_sharey=False)
        g.map_upper(sns.scatterplot, alpha=0.6, s=30)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=2)
        g.add_legend()
        plt.suptitle('Relationships Between Spending Features', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

# Function to perform spending analysis
def analyze_spending_patterns(df, target_var='Transported'):
    """Analyze spending patterns and their relationship with target"""
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    spending_cols = [col for col in spending_cols if col in df.columns]
    
    if len(spending_cols) > 0:
        # Total spending distribution
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['TotalSpending'].clip(upper=df['TotalSpending'].quantile(0.95)), kde=True)
        plt.title('Distribution of Total Spending (clipped at 95th percentile)', fontsize=14)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=target_var, y='TotalSpending', data=df)
        plt.title(f'Total Spending by {target_var}', fontsize=14)
        plt.ylim(0, df['TotalSpending'].quantile(0.95))
        
        plt.tight_layout()
        plt.show()
        
        # Spending by category
        plt.figure(figsize=(15, 10))
        melted_df = pd.melt(df, id_vars=[target_var], value_vars=spending_cols, 
                            var_name='Spending Category', value_name='Amount')
        
        # Remove extreme outliers for better visualization
        melted_df = melted_df[melted_df['Amount'] <= melted_df['Amount'].quantile(0.95)]
        
        sns.boxplot(x='Spending Category', y='Amount', hue=target_var, data=melted_df)
        plt.title('Spending by Category and Transport Status', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Zero vs non-zero spending analysis
    plt.figure(figsize=(14, 7))
    
    for i, col in enumerate(spending_cols, 1):
        df[f'{col}_Used'] = (df[col] > 0).astype(int)
    
    used_cols = [f'{col}_Used' for col in spending_cols]
    melted_df_used = pd.melt(df, id_vars=[target_var], value_vars=used_cols,
                             var_name='Service Used', value_name='Used (1=Yes)')
    
    sns.countplot(x='Service Used', hue=target_var, data=melted_df_used)
    plt.title('Service Usage by Transport Status', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to analyze group travel patterns
def analyze_group_patterns(df, target_var='Transported'):
    """Analyze patterns related to group travel"""
    if 'GroupSize' in df.columns and 'GroupId' in df.columns:
        plt.figure(figsize=(15, 5))
        
        # Distribution of group sizes
        plt.subplot(1, 3, 1)
        sns.countplot(x='GroupSize', data=df[df['GroupSize'] <= 8])  # Limit to common group sizes
        plt.title('Distribution of Group Sizes', fontsize=14)
        
        # Group size vs target
        plt.subplot(1, 3, 2)
        sns.countplot(x='GroupSize', hue=target_var, data=df[df['GroupSize'] <= 8])
        plt.title(f'Group Size vs {target_var}', fontsize=14)
        
        # Proportion transported by group size
        plt.subplot(1, 3, 3)
        prop_by_group = df.groupby('GroupSize')[target_var].mean()
        sns.barplot(x=prop_by_group.index[:8], y=prop_by_group.values[:8])
        plt.title(f'Proportion {target_var} by Group Size', fontsize=14)
        plt.ylabel(f'Proportion {target_var}')
        
        plt.tight_layout()
        plt.show()
        
        # Transportation consistency within groups
        group_stats = df.groupby('GroupId').agg({
            target_var: ['mean', 'count']
        })
        group_stats.columns = ['TransportProportion', 'GroupSize']
        
        # Calculate if group was wholly transported, partially transported, or not transported
        group_stats['TransportStatus'] = pd.cut(
            group_stats['TransportProportion'], 
            bins=[-0.01, 0.01, 0.99, 1.01],
            labels=['None Transported', 'Some Transported', 'All Transported']
        )
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='TransportStatus', data=group_stats)
        plt.title('Transportation Status by Group', fontsize=14)
        plt.ylabel('Number of Groups')
        plt.tight_layout()
        plt.show()