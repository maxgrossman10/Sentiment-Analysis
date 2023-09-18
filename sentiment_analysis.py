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

df["Review"] = df["Review"].apply(  # Apply function to each item
    lambda x: " ".join(  # Anonymous function with x as a review / join non-stop words into new list
        [
            word for word in x.split() if word.lower() not in stop
        ]  # iterate through list of words, eliminate stop words
    )
)

# Print df
df.head(20)


# %% LEMMATIZATION

# libraries
import nltk
from nltk.stem import WordNetLemmatizer

# Download wordnet database
nltk.download("wordnet")

# Perform lemmatization
lemmatizer = WordNetLemmatizer()  # Create instance of WNL() class

df["Review"] = df["Review"].apply(  # Apply below function to each item
    lambda x: " ".join(  # Anonymous lemmatize function for each review, then rejoin the list
        [lemmatizer.lemmatize(word) for word in x.split()]  # Apply lemmatizer
    )
)

# Show 20 entries
df.head(20)

# %% TOKENIZE & PAD

# Libraries
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# punkt NLTK package
nltk.download("punkt")

# Tokenize reviews
df["tokens"] = df["Review"].apply(word_tokenize)

# Convert tokens list into single list of text
text_list = df["tokens"].apply(" ".join).tolist()

# Initialize the tokenizer with a specified vocabulary size
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(text_list)

# Convert token lists to sequences
sequences = tokenizer.texts_to_sequences(text_list)

# Choose max sequence length
max_length = 14

# Pad sequences
padded_sequences = pad_sequences(
    sequences, maxlen=max_length, padding="post", truncating="post"
)

# Display df info and first 5 entries
print(df.info())
print(df["tokens"].head(5))

# Print the first padded sequence
print(padded_sequences[0])
