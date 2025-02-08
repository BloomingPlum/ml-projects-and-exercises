from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd

def drop_unnecessary_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (list): List of column names to drop.
        
    Returns:
        pd.DataFrame: DataFrame after dropping the specified columns.
    """
    # Remove the columns specified in columns_to_drop
    return df.drop(columns=columns_to_drop)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and validation sets.
    
    The split is stratified based on the 'Exited' column to ensure class balance.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: A tuple containing (train_df, val_df).
    """
    # Use stratified splitting based on the 'Exited' column to preserve class distribution
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Exited'])

def separate_inputs_targets(df, target_col='Exited'):
    """
    Separate the DataFrame into input features and target labels.
    
    It assumes that the first column is not an input feature and that the target column is specified.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column. Default is 'Exited'.
        
    Returns:
        tuple: A tuple containing (inputs DataFrame, targets Series).
    """
    # Create a list of columns to be used as inputs by excluding the first and last columns
    input_cols = list(df.columns)[1:-1]  # Exclude first and last column
    return df[input_cols].copy(), df[target_col].copy()

def identify_column_types(df):
    """
    Identify numeric and categorical columns in the DataFrame.
    
    The function excludes the first numeric column from the numeric columns list.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing only input features.
        
    Returns:
        tuple: A tuple containing a list of numeric column names and a list of categorical column names.
    """
    # Select numeric columns (ignoring the first column, which might be an ID or index)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[1:]
    # Select categorical columns (dtype 'object')
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def encode_categorical_features(df_train, df_val, categorical_cols):
    """
    Encode categorical features using OneHotEncoder for both training and validation sets.
    
    Parameters:
        df_train (pd.DataFrame): Training DataFrame containing categorical features.
        df_val (pd.DataFrame): Validation DataFrame containing categorical features.
        categorical_cols (list): List of categorical column names.
        
    Returns:
        tuple: A tuple containing:
            - Transformed training DataFrame with encoded features.
            - Transformed validation DataFrame with encoded features.
            - The fitted OneHotEncoder.
    """
    # Initialize OneHotEncoder; drop one column for binary features to avoid multicollinearity
    encoder = OneHotEncoder(drop='if_binary', sparse=False)
    # Fit the encoder on the training set's categorical columns
    encoder.fit(df_train[categorical_cols])
    
    # Retrieve the new column names after encoding
    encoded_cols = list(encoder.get_feature_names(categorical_cols))
    # Transform the training set categorical features and assign to new columns
    df_train[encoded_cols] = encoder.transform(df_train[categorical_cols])
    # Transform the validation set categorical features and assign to new columns
    df_val[encoded_cols] = encoder.transform(df_val[categorical_cols])
    
    # Drop the original categorical columns from both datasets
    return df_train.drop(columns=categorical_cols), df_val.drop(columns=categorical_cols), encoder

def scale_numeric_features(df_train, df_val, numeric_cols, scaler_numeric=True):
    """
    Scale numeric features using MinMaxScaler for both training and validation sets.
    
    Parameters:
        df_train (pd.DataFrame): Training DataFrame containing numeric features.
        df_val (pd.DataFrame): Validation DataFrame containing numeric features.
        numeric_cols (list): List of numeric column names.
        scaler_numeric (bool): Flag to indicate whether scaling should be applied.
        
    Returns:
        tuple: A tuple containing:
            - Transformed training DataFrame with scaled numeric features.
            - Transformed validation DataFrame with scaled numeric features.
            - The fitted MinMaxScaler, or None if scaling was not applied.
    """
    scaler = None  # Define scaler outside the if block to avoid UnboundLocalError

    if scaler_numeric:
        scaler = MinMaxScaler()
        # Fit the scaler using the training set numeric features
        scaler.fit(df_train[numeric_cols])
        # Apply the scaling to the training set
        df_train[numeric_cols] = scaler.transform(df_train[numeric_cols])
        # Apply the same scaling to the validation set
        df_val[numeric_cols] = scaler.transform(df_val[numeric_cols])

    # Always return scaler (either fitted or None)
    return df_train, df_val, scaler

def preprocess_data(raw_df, scaler_numeric=False):
    """
    Preprocess raw data by performing column dropping, splitting, encoding, and scaling.
    
    The function processes the raw DataFrame and returns a dictionary containing preprocessed training and
    validation inputs and targets, along with the fitted encoder and scaler.
    
    Parameters:
        raw_df (pd.DataFrame): The raw input DataFrame.
        scaler_numeric (bool): Flag to indicate whether numeric features should be scaled.
        
    Returns:
        dict: A dictionary with keys:
            - 'train_X': Preprocessed training input features.
            - 'train_y': Training target labels.
            - 'val_X': Preprocessed validation input features.
            - 'val_y': Validation target labels.
            - 'encoder': The fitted OneHotEncoder.
            - 'scaler': The fitted MinMaxScaler (or a default scaler if scaling was not applied).
    """
    # Drop columns that are not needed for analysis
    raw_df = drop_unnecessary_columns(raw_df, ['CustomerId', 'Surname'])
    # Split the data into training and validation sets with stratification on 'Exited'
    train_df, val_df = split_data(raw_df)

    # Separate the inputs and target variables for both training and validation sets
    train_inputs, train_targets = separate_inputs_targets(train_df)
    val_inputs, val_targets = separate_inputs_targets(val_df)

    # Identify which columns are numeric and which are categorical in the input features
    numeric_cols, categorical_cols = identify_column_types(train_inputs)

    # Encode the categorical features in both training and validation datasets
    train_inputs, val_inputs, encoder = encode_categorical_features(train_inputs, val_inputs, categorical_cols)

    # Scale the numeric features if scaler_numeric is True
    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols, scaler_numeric)

    # Ensure that scaler is defined to prevent UnboundLocalError
    if scaler is None:
        scaler = MinMaxScaler()  # Provides a default scaler if scaling was not applied

    # Return a dictionary with all preprocessed components
    return {
        'train_X': train_inputs,
        'train_y': train_targets,
        'val_X': val_inputs,
        'val_y': val_targets,
        'encoder': encoder,  # Return the fitted OneHotEncoder
        'scaler': scaler     # Return the fitted or default MinMaxScaler
    }

def preprocess_new_data(new_df, encoder, scaler, scaler_numeric=False):
    """
    Preprocess new (test) data using the trained encoder and scaler.

    Parameters:
        new_df (pd.DataFrame): The raw new/test DataFrame.
        encoder (OneHotEncoder): A pre-fitted OneHotEncoder for categorical features.
        scaler (MinMaxScaler): A pre-fitted MinMaxScaler for numeric features.
        scaler_numeric (bool): Whether to apply scaling to numeric columns (default: False).

    Returns:
        pd.DataFrame: A transformed DataFrame containing original numeric columns and encoded categorical features.
    """
    # Drop unnecessary columns if they exist; ignore errors if columns are not found
    columns_to_drop = ['id', 'CustomerId', 'Surname']
    new_df = new_df.drop(columns=columns_to_drop, errors='ignore') 

    # Identify categorical and numeric columns in the new data
    categorical_cols = new_df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = new_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Process categorical features using the pre-fitted encoder
    cat_encoded = encoder.transform(new_df[categorical_cols])
    # Retrieve the names for the encoded features
    encoded_feature_names = encoder.get_feature_names(categorical_cols)  
    # Create a DataFrame from the encoded features
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoded_feature_names, index=new_df.index)

    # Process numeric features: scale them if scaler_numeric is True
    if scaler_numeric:
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
    
    # Remove the original categorical columns as they have been encoded
    new_df = new_df.drop(columns=categorical_cols)

    # Concatenate the encoded categorical DataFrame with the remaining numeric features
    new_df = pd.concat([new_df, cat_encoded_df], axis=1)

    return new_df