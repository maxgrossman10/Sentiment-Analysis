# %% LIBRARIES & DF
# Libraries
import os
import pandas as pd
import seaborn as sn

# Define the working directory path
data = pd.read_csv("C:\\Projects\\Data\\reviews_data.csv")
df = data[["Review", "Rating"]].dropna()

# Inspect the data types / count null values
df.info()
df.isna().sum()


###############################################################################################
# %% MAKE LOWERCASE | REMOVE EMOTICONS, PUNCTUATION, NON-ENGLISH

# Make all lowercase
df["Review"] = df["Review"].str.lower()

# Remove punctuation from the 'review' column using str.replace
df["Review"] = df["Review"].str.replace(r"[^\w\s]", "", regex=True)
# ^ inside the set means "anything but" the characters listed in the set
# \w is for any word character
# \s is for any whitespace character

# Display the df
print(df.head(20))


###############################################################################################
# %% ELIMINATE STOP WORDS

# Libraries
import nltk
from nltk.corpus import stopwords

# Download pre-defined stop words library
nltk.download("stopwords")

# Get the English stop words from library and make a list with set()
stop = set(stopwords.words("english"))

df["review"] = df["review"].apply(  # Apply function to each item
    lambda x: " ".join(  # Anonymous function with x as a review / join non-stop words into new list
        [
            word for word in x.split() if word.lower() not in stop
        ]  # iterate through list of words, eliminate stop words
    )
)

# Print df
df.head(20)
