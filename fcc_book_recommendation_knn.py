import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import urllib.request
import zipfile


url = 'https://cdn.freecodecamp.org/project-data/books/book-crossings.zip'
output_file = 'book-crossings.zip'

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(output_file, 'wb') as out_file:
    out_file.write(response.read())

with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall('.')


books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'


df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})


user_counts = df_ratings['user'].value_counts()
df_ratings_filtered = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 180].index)]


book_counts = df_ratings_filtered['isbn'].value_counts()
df_ratings_filtered = df_ratings_filtered[df_ratings_filtered['isbn'].isin(book_counts[book_counts >= 80].index)]


df_merged = pd.merge(df_ratings_filtered, df_books, on='isbn')


book_user_matrix = df_merged.pivot_table(index='title', columns='user', values='rating')
book_user_matrix.fillna(0, inplace=True)


book_user_sparse = csr_matrix(book_user_matrix.values)


knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(book_user_sparse)


#
def get_recommends(book=""):
    if book not in book_user_matrix.index:
        print(f"Book '{book}' not found in index!")
        return [book, []]

    book_idx = book_user_matrix.index.get_loc(book)
    distances, indices = knn_model.kneighbors(book_user_matrix.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6)
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        title = book_user_matrix.index[indices.flatten()[i]]
        distance = distances.flatten()[i]
        recommended_books.append([title, distance])
    return [book, recommended_books]



books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)



def test_book_recommendation():
    try:
        recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
        if type(recommends) == list and type(recommends[1]) == list and len(recommends[1]) >= 4:
            print("You passed the challenge! 🎉🎉🎉🎉🎉")
        else:
            print("You haven't passed yet. Keep trying!")
    except:
        print("You haven't passed yet. Keep trying!")

test_book_recommendation()
