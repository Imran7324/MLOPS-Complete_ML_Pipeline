import os,string
import logging
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
#Logging Module Code for Data Preprocessing
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('Data_Preprocessing')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path=os.path.join(log_dir,'Data_Preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# Function for Transforming the text
def transform_text(text):
    ps=PorterStemmer()
    text=text.lower()# To lower case
    text=nltk.word_tokenize(text)# Tokenization of the text
    text=[word for word in text if word.isalnum()] #Return the non-alphanumeric words
    # Removes the stopword and punctuations from the text
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #Stemming the text
    text=[ps.stem(word) for word in text]
    return " ".join(text)
# Code for Preprocessing the data
def preprocess_data(df,text_column='text',target_column='target'):
    try:
        logger.debug("Preprocessing is initiated for the DataFrame:")
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target Column is Encoded")
        df=df.drop_duplicates(keep='first')
        logger.debug("Duplicates Removed")
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug("Text column has been transformed")
        return df
    except KeyError as e:
        logger.error("Column Not Found:%s",e)
        raise
    except Exception as e:
        logger.error("Unknown Error: %s", e)
        raise
# Main funtion to Load data, preprocess it and save it
def main(text_column="text",target_column="target"):
    try:
        train_data=pd.read_csv("./data/raw/train.csv")
        test_data=pd.read_csv("./data/raw/test.csv")
        logger.debug("Data Loaded Properly")
        train_processed_data=preprocess_data(train_data,text_column,target_column)
        test_processed_data=preprocess_data(test_data,text_column,target_column)
        #Storing the processed data to specific csv files
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path,"training_processed_data.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path,"testing_processed_data.csv"),index=False)
        logger.debug("Process data is saved to %s", data_path)
    except FileNotFoundError as e:
        logger.error("File Not found %s",e)
    except pd.errors.EmptyDataError as e:
        logger.error("No Data %s",e)
    except Exception as e:
        logger.error("Faled %s",e)
        print(f"Error:{e}")
if __name__=="__main__":
    main()
    
    
    