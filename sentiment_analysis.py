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

###############################################################################################
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


###############################################################################################
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


###############################################################################################
# %% CHOOSE MAXIMUM SEQUENCE LENGTH

# Library
import matplotlib.pyplot as plt

# Count number of words in each review and store in doc_length
df["doc_length"] = df["Review"].apply(lambda x: len(x.split()))

# Plot the distribution of doc_length
plt.hist(df["doc_length"], bins=1000)
plt.xlabel("Document Length")
plt.ylabel("Frequency")
plt.title("Distribution of Document Lengths")

# Limit x-axis from 0 to 100
plt.xlim(0, 50)

# Print graph
plt.show()

# Calc mean & 95th percentile
mean_length = df["doc_length"].mean()
percentile_95 = df["doc_length"].quantile(0.95)

# Print results
print(f"Mean Document Length: {mean_length}")
print(f"95th Percentile Document Length: {percentile_95}")


###############################################################################################
# %% VECTORIZE

# Libraries
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the tokenized reviews
vectorizer = TfidfVectorizer(max_features=1000)  # Only consider top 1000 words
X = vectorizer.fit_transform(df["Review"])

# Display the features and matrix shape
print(vectorizer.get_feature_names_out())
print(X.shape)

###############################################################################################
# %% SPLITTING DATA

# Library
from sklearn.model_selection import train_test_split

# 'rating' column is target variable
y = df["Rating"]

# Training-Test-Validation split 70-15-15
X_train, X_remain, y_train, y_remain = train_test_split(
    padded_sequences, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_remain, y_remain, test_size=0.5, random_state=42, stratify=y_remain
)

###############################################################################################
# %% LSTM MODEL WITH EARLY STOPPING CRITERIA

"""
EarlyStopping is imported from tensorflow.keras.callbacks and instantiated with monitor='val_loss'. 
Patience=3 means if the validation loss doesn't improve for 3 consecutive epochs, the training process will stop.
"""

# Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Max vocab size for tokenizer / num of dimensions
vocab_size = 10000
embedding_dimensions = 16

# Create sequential model and the embedding, LSTM, and dense layers
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dimensions,
            input_length=max_length,
        ),
        LSTM(32),
        Dense(1, activation="sigmoid"),
    ]
)

# Use binary-crossentropy for loss function and Adam optimizer
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Define early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

# Train the model with early stopping callback
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

###############################################################################################
# %% VISUALIZING MODE TRAINING PROCESS

# Library
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Plot training & validation loss values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

###############################################################################################
# %% scrap
# Import library
import numpy as np

# Convert padded_sequences to DataFrame
df_padded_sequences = pd.DataFrame(padded_sequences)

# Concatenate df and df_padded_sequences along columns
df_preprocessed = pd.concat([df.reset_index(drop=True), df_padded_sequences], axis=1)

# Remove 'review' and 'tokens' columns as they are raw and tokenized reviews respectively
df_preprocessed.drop(columns=["Review", "tokens"], inplace=True)

# # Export to CSV
# output_path = "C:\\Projects\\WGU\\D213\\Python"
# df_preprocessed.to_csv(os.path.join(output_path, "prepared_dataset.csv"), index=False)
