```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")
links = pd.read_csv("links.csv")
```


```python
print("movies:", movies.shape)
print("ratings:", ratings.shape)
print("tags:", tags.shape)
print("links:", links.shape)
```

    movies: (87585, 3)
    ratings: (32000204, 4)
    tags: (2000072, 4)
    links: (87585, 3)



```python
#选择一个用户，看评分过多少电影
user_id = 1

user_ratings = ratings[ratings["userId"] == user_id].copy()

print("User:", user_id)
print("Number of rated movies:", len(user_ratings))
```

    User: 1
    Number of rated movies: 141



```python
#把电影信息的表合并起来
user_ratings = user_ratings.merge(
    movies,
    on="movieId",
    how="left"
)
user_ratings[
    ["title", "genres", "rating"]
].sort_values(
    "rating",
    ascending=False
).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>Platoon (1986)</td>
      <td>Drama|War</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>North by Northwest (1959)</td>
      <td>Action|Adventure|Mystery|Romance|Thriller</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Sabrina (1954)</td>
      <td>Comedy|Romance</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Dangerous Liaisons (1988)</td>
      <td>Drama|Romance</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Citizen Kane (1941)</td>
      <td>Drama|Mystery</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>All About Eve (1950)</td>
      <td>Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Women, The (1939)</td>
      <td>Comedy</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>To Catch a Thief (1955)</td>
      <td>Crime|Mystery|Romance|Thriller</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Patton (1970)</td>
      <td>Drama|War</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Secrets &amp; Lies (1996)</td>
      <td>Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Jude (1996)</td>
      <td>Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>Shakespeare in Love (1998)</td>
      <td>Comedy|Drama|Romance</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Back to the Future (1985)</td>
      <td>Adventure|Comedy|Sci-Fi</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Deer Hunter, The (1978)</td>
      <td>Drama|War</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Elizabeth (1998)</td>
      <td>Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>American History X (1998)</td>
      <td>Crime|Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>Elephant Man, The (1980)</td>
      <td>Drama</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Graduate, The (1967)</td>
      <td>Comedy|Drama|Romance</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Lolita (1997)</td>
      <td>Drama|Romance</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Star Wars: Episode V - The Empire Strikes Back...</td>
      <td>Action|Adventure|Sci-Fi</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#找出用户1喜欢的电影类型
##4分以上的电影
liked_movies = user_ratings[
    user_ratings["rating"] >= 4
].copy()
print("Liked movies:", len(liked_movies))
```

    Liked movies: 83



```python
#把不同的genre拆开
genre_list = []

for genres in liked_movies["genres"].dropna():
    genre_list.extend(genres.split("|"))
```


```python
genre_counts = pd.Series(genre_list).value_counts()

genre_counts
```




    Drama        63
    Comedy       28
    Romance      20
    Action       14
    War          12
    Sci-Fi       11
    Adventure    11
    Crime        10
    Thriller      9
    Mystery       6
    Fantasy       2
    Children      1
    Western       1
    Horror        1
    Film-Noir     1
    Name: count, dtype: int64




```python
#每个 Genre 对应电影的平均评分
genre_scores = {}

for genre in genre_counts.index:
    
    genre_movies = liked_movies[
        liked_movies["genres"].fillna("").str.contains(
            genre,
            regex=False
        )
    ]
    
    genre_scores[genre] = genre_movies["rating"].mean()
```


```python
genre_scores = pd.Series(genre_scores).sort_values(
    ascending=False
)

genre_scores
```




    Crime        5.000000
    Children     5.000000
    Horror       5.000000
    Film-Noir    5.000000
    Sci-Fi       4.909091
    Mystery      4.833333
    Thriller     4.777778
    Action       4.714286
    Drama        4.650794
    Adventure    4.636364
    Comedy       4.607143
    Romance      4.600000
    War          4.583333
    Fantasy      4.500000
    Western      4.000000
    dtype: float64




```python
#找出用户偏好的genre
genre_preference = pd.DataFrame({
    "count": genre_counts,
    "avg_rating": genre_scores
})

genre_preference["score"] = (
    genre_preference["count"] *
    genre_preference["avg_rating"]
)

genre_preference = genre_preference.sort_values(
    "score",
    ascending=False
)

genre_preference

genre_preference.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>avg_rating</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Drama</th>
      <td>63</td>
      <td>4.650794</td>
      <td>293.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>28</td>
      <td>4.607143</td>
      <td>129.0</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>20</td>
      <td>4.600000</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>Action</th>
      <td>14</td>
      <td>4.714286</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>War</th>
      <td>12</td>
      <td>4.583333</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>11</td>
      <td>4.909091</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>11</td>
      <td>4.636364</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>10</td>
      <td>5.000000</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>9</td>
      <td>4.777778</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>Mystery</th>
      <td>6</td>
      <td>4.833333</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#定义用户最喜欢的genre
favorite_genres = genre_preference.head(5).index.tolist()

favorite_genres
```




    ['Drama', 'Comedy', 'Romance', 'Action', 'War']




```python
#找出用户没看过的电影
watched_movies = set(
    user_ratings["movieId"]
)
candidate_movies = movies[
    ~movies["movieId"].isin(watched_movies)
].copy()
print("Total movies:", len(movies))
print("Watched movies:", len(watched_movies))
print("Candidate movies:", len(candidate_movies))
```

    Total movies: 87585
    Watched movies: 141
    Candidate movies: 87444



```python
#给候选电影算分
def genre_score(genres):
    if pd.isna(genres):
        return 0
    
    movie_genres = set(genres.split("|"))
    
    return len(
        movie_genres.intersection(favorite_genres)
    )
candidate_movies["genre_score"] = (
    candidate_movies["genres"]
    .apply(genre_score)
) 
candidate_movies["genre_score"].value_counts().sort_index()
#去掉等于0的电影
candidate_movies = candidate_movies[
    candidate_movies["genre_score"] > 0
].copy()
```


```python
#计算每部电影评分
movie_stats = ratings.groupby("movieId").agg(
    avg_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()
movie_stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>avg_rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3.897438</td>
      <td>68997</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3.275758</td>
      <td>28904</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3.139447</td>
      <td>13134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2.845331</td>
      <td>2806</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3.059602</td>
      <td>13154</td>
    </tr>
  </tbody>
</table>
</div>




```python
recommendations = candidate_movies.merge(
    movie_stats,
    on="movieId",
    how="left"
)
print(recommendations.shape)
print(recommendations.columns.tolist())
recommendations.head()
```

    (57810, 6)
    ['movieId', 'title', 'genres', 'genre_score', 'avg_rating', 'rating_count']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>genre_score</th>
      <th>avg_rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1</td>
      <td>3.897438</td>
      <td>68997.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>2</td>
      <td>3.139447</td>
      <td>13134.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>3</td>
      <td>2.845331</td>
      <td>2806.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>1</td>
      <td>3.059602</td>
      <td>13154.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
      <td>1</td>
      <td>3.868277</td>
      <td>29490.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#丢掉没有评分的电影
recommendations = recommendations.dropna(
    subset=["avg_rating", "rating_count"]
)
```


```python
#计算所有电影平均分
global_mean = movie_stats["avg_rating"].mean()

C = 50

recommendations["weighted_rating"] = (
    recommendations["rating_count"] /
    (recommendations["rating_count"] + C)
    * recommendations["avg_rating"]
    +
    C /
    (recommendations["rating_count"] + C)
    * global_mean
)
```


```python
#用户喜欢程度60%+电影本身质量40%
##（最终得分是一个简单的启发式排序函数，而不是一个经过学习训练的推荐模型。）
recommendations["final_score"] = (
    0.6 * recommendations["genre_score"]
    +
    0.4 * recommendations["weighted_rating"]
)
```


```python
recommendations = recommendations.sort_values(
    "final_score",
    ascending=False
)
```


```python
#推荐前二十
top20 = recommendations[
    [
        "title",
        "genres",
        "genre_score",
        "avg_rating",
        "rating_count",
        "weighted_rating",
        "final_score"
    ]
].head(20)

top20
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>genre_score</th>
      <th>avg_rating</th>
      <th>rating_count</th>
      <th>weighted_rating</th>
      <th>final_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7057</th>
      <td>White Sun of the Desert, The (Beloe solntse pu...</td>
      <td>Action|Adventure|Comedy|Drama|Romance|War</td>
      <td>5</td>
      <td>3.738220</td>
      <td>191.0</td>
      <td>3.586117</td>
      <td>4.434447</td>
    </tr>
    <tr>
      <th>13659</th>
      <td>Lazybones (1925)</td>
      <td>Action|Comedy|Drama|Romance|War</td>
      <td>5</td>
      <td>2.500000</td>
      <td>8.0</td>
      <td>2.935416</td>
      <td>4.174166</td>
    </tr>
    <tr>
      <th>964</th>
      <td>Henry V (1989)</td>
      <td>Action|Drama|Romance|War</td>
      <td>4</td>
      <td>4.100955</td>
      <td>4819.0</td>
      <td>4.089701</td>
      <td>4.035880</td>
    </tr>
    <tr>
      <th>2073</th>
      <td>Spartacus (1960)</td>
      <td>Action|Drama|Romance|War</td>
      <td>4</td>
      <td>3.826786</td>
      <td>6033.0</td>
      <td>3.820032</td>
      <td>3.928013</td>
    </tr>
    <tr>
      <th>8510</th>
      <td>Life is a Miracle (Zivot je cudo) (2004)</td>
      <td>Comedy|Drama|Musical|Romance|War</td>
      <td>4</td>
      <td>3.844311</td>
      <td>167.0</td>
      <td>3.650941</td>
      <td>3.860376</td>
    </tr>
    <tr>
      <th>3872</th>
      <td>Stunt Man, The (1980)</td>
      <td>Action|Adventure|Comedy|Drama|Romance|Thriller</td>
      <td>4</td>
      <td>3.693017</td>
      <td>759.0</td>
      <td>3.650500</td>
      <td>3.860200</td>
    </tr>
    <tr>
      <th>1485</th>
      <td>Wings (1927)</td>
      <td>Action|Drama|Romance|War</td>
      <td>4</td>
      <td>3.684864</td>
      <td>403.0</td>
      <td>3.609832</td>
      <td>3.843933</td>
    </tr>
    <tr>
      <th>3753</th>
      <td>Operation Petticoat (1959)</td>
      <td>Action|Comedy|Romance|War</td>
      <td>4</td>
      <td>3.637399</td>
      <td>746.0</td>
      <td>3.597681</td>
      <td>3.839072</td>
    </tr>
    <tr>
      <th>7482</th>
      <td>Revolutionary Girl Utena: Adolescence of Utena...</td>
      <td>Action|Adventure|Animation|Comedy|Drama|Fantas...</td>
      <td>4</td>
      <td>3.983051</td>
      <td>59.0</td>
      <td>3.534441</td>
      <td>3.813777</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Rob Roy (1995)</td>
      <td>Action|Drama|Romance|War</td>
      <td>4</td>
      <td>3.529142</td>
      <td>12868.0</td>
      <td>3.527114</td>
      <td>3.810845</td>
    </tr>
    <tr>
      <th>9129</th>
      <td>Tiger and the Snow, The (La tigre e la neve) (...</td>
      <td>Comedy|Drama|Romance|War</td>
      <td>4</td>
      <td>3.666667</td>
      <td>171.0</td>
      <td>3.516987</td>
      <td>3.806795</td>
    </tr>
    <tr>
      <th>31003</th>
      <td>Allied (2016)</td>
      <td>Action|Drama|Romance|Thriller|War</td>
      <td>4</td>
      <td>3.529502</td>
      <td>983.0</td>
      <td>3.504118</td>
      <td>3.801647</td>
    </tr>
    <tr>
      <th>8107</th>
      <td>Kingdom of Heaven (2005)</td>
      <td>Action|Drama|Romance|War</td>
      <td>4</td>
      <td>3.492898</td>
      <td>5210.0</td>
      <td>3.488261</td>
      <td>3.795304</td>
    </tr>
    <tr>
      <th>9603</th>
      <td>Great War, The (Grande guerra, La) (1959)</td>
      <td>Action|Adventure|Comedy|Drama|War</td>
      <td>4</td>
      <td>4.037500</td>
      <td>40.0</td>
      <td>3.463935</td>
      <td>3.785574</td>
    </tr>
    <tr>
      <th>9886</th>
      <td>I Served the King of England (Obsluhoval jsem ...</td>
      <td>Comedy|Drama|Romance|War</td>
      <td>4</td>
      <td>3.613014</td>
      <td>146.0</td>
      <td>3.457929</td>
      <td>3.783172</td>
    </tr>
    <tr>
      <th>2291</th>
      <td>Train of Life (Train de vie) (1998)</td>
      <td>Comedy|Drama|Romance|War</td>
      <td>4</td>
      <td>3.506977</td>
      <td>215.0</td>
      <td>3.412280</td>
      <td>3.764912</td>
    </tr>
    <tr>
      <th>10382</th>
      <td>Absolute Giganten (1999)</td>
      <td>Action|Comedy|Drama|Romance</td>
      <td>4</td>
      <td>3.818182</td>
      <td>44.0</td>
      <td>3.385682</td>
      <td>3.754273</td>
    </tr>
    <tr>
      <th>9280</th>
      <td>Girl Shy (1924)</td>
      <td>Action|Comedy|Drama|Romance</td>
      <td>4</td>
      <td>3.743590</td>
      <td>39.0</td>
      <td>3.328698</td>
      <td>3.731479</td>
    </tr>
    <tr>
      <th>7827</th>
      <td>El Cid (1961)</td>
      <td>Action|Adventure|Drama|Romance|War</td>
      <td>4</td>
      <td>3.407960</td>
      <td>201.0</td>
      <td>3.327706</td>
      <td>3.731082</td>
    </tr>
    <tr>
      <th>32394</th>
      <td>Their Finest (2017)</td>
      <td>Comedy|Drama|Romance|War</td>
      <td>4</td>
      <td>3.416129</td>
      <td>155.0</td>
      <td>3.315874</td>
      <td>3.726350</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
