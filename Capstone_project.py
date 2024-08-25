import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC

# Random seed for reproducibility
np.random.seed(10206068)
random.seed(10206068)

# Load the dataset
csv_file = 'spotify52kData.csv'
df = pd.read_csv(csv_file)

# Find duplicate rows
duplicate_rows = df[df.duplicated()]


# Output files for processed data
output_csv = 'output.csv'
dups_csv = 'dups.csv'
output_ignore_genre = 'output_ig.csv'
dups_ignore_g = 'dups_ig.csv'


# These are all of the columns that we can consider for the analysis
columns_to_consider = ['artists', 'album_name',	'track_name', 'popularity',	'duration',	'explicit',
                       'danceability', 'energy', 'key',	'loudness',	'mode',	'speechiness',	'acousticness',
                       'instrumentalness',	'liveness',	'valence', 'tempo',	'time_signature', 'track_genre']


# Same as above but does not include the track_genre column
col_to_consider_2 = ['artists', 'album_name',	'track_name', 'popularity',	'duration',	'explicit',
                       'danceability', 'energy', 'key',	'loudness',	'mode',	'speechiness',	'acousticness',
                       'instrumentalness',	'liveness',	'valence', 'tempo',	'time_signature']

# Handling duplicates considering all columns
dups = df[df.duplicated(subset=columns_to_consider, keep=False)]
df_no_dups = df.drop_duplicates(subset=columns_to_consider, keep='first')

# Handling duplicates but ignore track_genre column (ig = ignore genre)
dups_ignore_genre = df[df.duplicated(subset=col_to_consider_2, keep=False)]
df_no_dups_ig = df.drop_duplicates(subset=col_to_consider_2, keep='first')


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 20)

#Save processed dataframes to CSV files
df_no_dups.to_csv(output_csv, index=False)
dups.to_csv(dups_csv, index=False)

df_no_dups_ig.to_csv(output_ignore_genre, index=False)
dups_ignore_genre.to_csv(dups_ignore_g, index=False)

# Output number of records after removing duplicates
print("Number of records no duplicates with genre difference: " + str(len(df_no_dups_ig)))

print("Question 1: Song length and popularity")

# Remove records that have missing values for duration and popularity
eda_data = df_no_dups_ig.dropna(subset=['duration', 'popularity'])


duration_popularity_corr, p_value_corr = spearmanr(eda_data['duration'], eda_data['popularity'])


plt.figure(figsize=(12, 6))

# Plot the distribution of song duration
plt.subplot(1, 2, 1)
sns.histplot(eda_data['duration'], kde=True)
plt.title('Distribution of Song Duration')
plt.xlabel('Duration (ms)')

# Plot the distribution of song popularity
plt.subplot(1, 2, 2)
sns.histplot(eda_data['popularity'], kde=True)
plt.title('Distribution of Song Popularity')
plt.xlabel('Popularity Score')

plt.tight_layout()
plt.show()

print(f"The correlation coefficient between duration and popularity: {duration_popularity_corr:.4f}")
print(f"P-value: {p_value_corr}")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=eda_data, x='duration', y='popularity', alpha=0.7)
plt.title('Scatter Plot of Song Duration vs Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity Score')
plt.show()

print("Question 2:")

#Separate popularity data based on explicit content
explicit_popularity = df_no_dups_ig[df_no_dups_ig['explicit'] == True]['popularity']
non_explicit_popularity = df_no_dups_ig[df_no_dups_ig['explicit'] == False]['popularity']

# Calculating mean popularity for explicit and non-explicit songs
explicit_mean = explicit_popularity.mean()
non_explicit_mean = non_explicit_popularity.mean()

print(f"Average popularity of explicit songs: {explicit_mean:.2f}")
print(f"Average popularity of non-explicit songs: {non_explicit_mean:.2f}")




plt.figure(figsize=(12, 6))

#Histogram explicit songs
plt.subplot(1, 2, 1)
sns.histplot(explicit_popularity, kde=True, bins=50)
plt.title('Popularity Distribution of Explicit Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

#Histogram for non-explicit songs
plt.subplot(1, 2, 2)
sns.histplot(non_explicit_popularity, kde=True, bins=50)
plt.title('Popularity Distribution of Non-Explicit Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#Conduct Mann_Whitney U test to compare popularity distribution
u_stat_2, p_value_2 = mannwhitneyu(explicit_popularity, non_explicit_popularity)

print("Mann-Whitney U test statistic: " + str(u_stat_2) + ", p-value: " + str(p_value_2))

# Calculate standard deviation
std_explicit = np.std(explicit_popularity, ddof=1)
std_non_explicit = np.std(non_explicit_popularity, ddof=1)

#Cohen's d to measure the effect size
cohen_d_2 = (explicit_mean - non_explicit_mean) / np.sqrt((std_explicit ** 2 + std_non_explicit ** 2) / 2)


print(f"Cohen's d: {cohen_d_2}")

# Bootstrap confidence interval for the mean difference
def bootstrap_mean_diff(data1, data2, n_bootstrap=50000):
    bootstrap_diffs = []
    for i in range(n_bootstrap):
        bootstrap_sample1 = np.random.choice(data1, size=len(data1), replace=True)
        bootstrap_sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrap_diffs.append(np.mean(bootstrap_sample1) - np.mean(bootstrap_sample2))
    lower, upper = np.percentile(bootstrap_diffs, [2.5, 97.5])
    return lower, upper

# Calculating the bootstrap confidence interval
bootstrap_ci = bootstrap_mean_diff(explicit_popularity, non_explicit_popularity)

print(f"Bootstrap confidence interval for the mean difference: " + str(bootstrap_ci))


print("Question 3:")

#Separate songs based on song mode looking at popularity
major_key_popularity = df_no_dups_ig[df_no_dups_ig['mode'] == 1]['popularity']
minor_key_popularity = df_no_dups_ig[df_no_dups_ig['mode'] == 0]['popularity']

average_popularity_major = major_key_popularity.mean()
average_popularity_minor = minor_key_popularity.mean()

print("Average popularity of major key songs: " + str(average_popularity_major))
print("Average popularity of minor key songs: " + str(average_popularity_minor))


plt.figure(figsize=(12, 6))

# Histogram major key songs
plt.subplot(1, 2, 1)
sns.histplot(major_key_popularity, kde=True, bins=50)
plt.title('Popularity Distribution of Major Key Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

#Histogram for minor key songs
plt.subplot(1, 2, 2)
sns.histplot(minor_key_popularity, kde=True, bins=50)
plt.title('Popularity Distribution of Minor Key Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Conducting the Mann-Whitney U test to compare popularity distributions
u_stat_3, p_value_3 = mannwhitneyu(major_key_popularity, minor_key_popularity)

cohen_d_3 = (average_popularity_major - average_popularity_minor) / df_no_dups_ig['popularity'].std(ddof=1)

print("Mann-Whitney U test statistic: " + str(u_stat_3) + ", p-value: " + str(p_value_3))
print("Cohen's d is: " + str(cohen_d_3))

print("Question 4")

features_to_analyze = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo']
target_variable = 'popularity'
linear_regression_results = {}


# Loop through each feature to analyze its relationship with popularity
for feature in features_to_analyze:

    X_feature = df_no_dups_ig[[feature]]
    y_popularity = df_no_dups_ig[target_variable]

    x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(X_feature, y_popularity, test_size = 0.15, random_state=10206068)

    # Initialize and train the Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(x_train_4, y_train_4)

    # Make predictions on the test set
    y_pred_4 = linear_model.predict(x_test_4)

    # Calculate R-squared and RMSE for the model
    r_squared_4 = linear_model.score(x_test_4, y_test_4)
    rmse_4 = np.sqrt(mean_squared_error(y_test_4, y_pred_4))

    linear_regression_results[feature] = {'R_Squared': r_squared_4, 'RMSE': rmse_4}

for feature, result in linear_regression_results.items():
    print("Feature: " + str(feature) + ", R^2: " + str(result['R_Squared']) + ", RMSE: " + str(result['RMSE']))

print("Question 5")

features_regression = df_no_dups_ig[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo']]
target_popularity = df_no_dups_ig['popularity']

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(features_regression, target_popularity,
                                                                    test_size=0.15, random_state=10206068)
# Initializing and fitting a Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(x_train_reg, y_train_reg)
y_pred_linear_reg = linear_reg_model.predict(x_test_reg)

# Evaluating the Linear Regression model performance
print("Linear Regression Model Performance")
print("R-squared:", r2_score(y_test_reg, y_pred_linear_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_linear_reg)))

ridge_reg_model = Ridge().fit(x_train_reg, y_train_reg)
lasso_reg_model = Lasso().fit(x_train_reg, y_train_reg)

y_pred_ridge_reg = ridge_reg_model.predict(x_test_reg)
y_pred_lasso_reg = lasso_reg_model.predict(x_test_reg)

print("Ridge Regression")
print("R²:", r2_score(y_test_reg, y_pred_ridge_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge_reg)))

print("Lasso Regression")
print("R²:", r2_score(y_test_reg, y_pred_lasso_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_lasso_reg)))

print("Question 6")

features_for_pca = df_no_dups_ig[features_to_analyze]

# Standardizing the features before applying PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_pca)

# Performing PCA to reduce dimensionality
pca = PCA()
features_pca = pca.fit_transform(features_scaled)

# Extracting explained variance ratios and calculating cumulative variance
explained_variance_ratios = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratios.cumsum()

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()

# Selecting the number of principal components that explain the majority of variance
n_components = 4
features_pca_reduced = features_pca[:, :n_components]

inertia_values = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=10206068)
    kmeans.fit(features_pca_reduced)
    inertia_values.append(kmeans.inertia_)


# Plotting the Elbow Method to determine the optimal number of clusters
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Make sure it is a copy
df_no_dups_ig = df_no_dups_ig.copy()

# Based on the Elbow plot, selecting the optimal number of clusters
n_clusters_optimal = 3
kmeans_final = KMeans(n_clusters=n_clusters_optimal, random_state=10206068)
cluster_labels = kmeans_final.fit_predict(features_pca_reduced)

# Assigning cluster labels to the original DataFrame
df_no_dups_ig['cluster'] = cluster_labels
genre_cluster_correspondence = pd.crosstab(df_no_dups_ig['cluster'], df_no_dups_ig['track_genre'])

print(genre_cluster_correspondence)

print('Question 7')

x_7 = df_no_dups_ig[['valence']]
y_7 = df_no_dups_ig['mode']

# Standardizing the feature to have a mean of 0 and a standard deviation of 1
scalar_7 = StandardScaler()
x_scaled_7 = scalar_7.fit_transform(x_7)

# Splitting the data into training and testing sets (85% train, 15% test)
x_train_7, x_test_7, y_train_7, y_test_7 = train_test_split(x_scaled_7, y_7, test_size=0.15, random_state=10206068)

# Initializing and training a Logistic Regression model with balanced class weights
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(x_train_7, y_train_7)

# Initializing and training a Support Vector Machine (SVM) model with balanced class weights
svm = SVC(probability=True, class_weight='balanced')
svm.fit(x_train_7, y_train_7)

# Making predictions on the test set using both models
y_pred_logreg = logreg.predict(x_test_7)
y_pred_svm = svm.predict(x_test_7)

print("Logistic Regression classification report:")
print(classification_report(y_test_7, y_pred_logreg))

print("Support Vector Machine classification report:")
print(classification_report(y_test_7, y_pred_svm))

accuracy_logreg = accuracy_score(y_test_7, y_pred_logreg)
accuracy_svm = accuracy_score(y_test_7, y_pred_svm)

print("Logistic Regression accuracy: " + str(accuracy_logreg))
print("Support Vector Machine accuracy: " + str(accuracy_svm))

print('Question 8')

features_nn = features_pca_reduced

label_encoder = LabelEncoder()
target_genre_encoded = label_encoder.fit_transform(df_no_dups_ig['track_genre'])
target_genre_categorical = to_categorical(target_genre_encoded)


x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(features_nn, target_genre_categorical, test_size=0.15, random_state=10206068)

# Building a Sequential Neural Network model
nn_model = Sequential()

# Input layer with the shape matching the number of features
nn_model.add(Input(shape=(x_train_nn.shape[1],)))

# Adding Dense layers with ReLU activation
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(32, activation='relu'))


nn_model.add(Dense(target_genre_categorical.shape[1], activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.summary()

optimizer = Adam(learning_rate=0.0005)
nn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training the neural network
nn_model.fit(x_train_nn, y_train_nn, epochs=10, batch_size=32, validation_split=0.1)

nn_loss, nn_accuracy = nn_model.evaluate(x_test_nn, y_test_nn)
print(f"Neural Network Test Accuracy: {nn_accuracy*100}")

print("Question 9")

# Loading the star ratings data
star_ratings = pd.read_csv("starRatings.csv", header=None)

# Selecting popularity scores for the first 5000 songs (matching the star ratings data)
popularity_scores = df['popularity'].iloc[:5000]

avg_star_ratings = star_ratings.mean(axis=0)

# Calculating the correlation between popularity scores and average star ratings
correlation_9 = popularity_scores.corr(avg_star_ratings)

# Identifying the top 10 most popular songs (greatest hits) based on popularity
greatest_hits = df.iloc[:5000].sort_values(by='popularity', ascending=False).head(10)

print("Correlation of popularity and average star ratings: " + str(correlation_9))
print("Greatest Hits based on popularity: ")
print(greatest_hits[['songNumber', 'artists', 'track_name', 'artists', 'popularity', 'track_genre']])

print("Question 10")

# Convert star ratings DataFrame to a numpy matrix for processing
ratings_matrix = star_ratings.values

# Function to get top-rated song recommendations for each user
def get_top_user_ratings(ratings, num_recommendations=10):
    top_recommendations = {}
    for user_index in range(ratings.shape[0]):
        user_ratings = ratings[user_index, :]
        top_rated_songs_indices = np.argsort(-user_ratings)

        filtered_recommendations = [index for index in top_rated_songs_indices if
                                    user_ratings[index] > 3][:num_recommendations]

        top_recommendations[user_index] = filtered_recommendations
    return top_recommendations

# Function to fill user recommendations with overall top-rated songs if fewer than 10 recommendations are found
def fill_with_overall_top_rated(user_recommendations, overall_top_rated, num_recommendations=10):
    for user_index, recommendations in user_recommendations.items():
        if len(recommendations) < num_recommendations:
            needed = num_recommendations - len(recommendations)
            recommendations.extend(overall_top_rated[:needed])
    return user_recommendations

# Get the top recommendations for each user based on their ratings
top_user_recommendations = get_top_user_ratings(ratings_matrix)

# Identify the top 10 overall highest-rated songs across all users
overall_top_rated_indices = np.argsort(-np.nanmean(ratings_matrix, axis=0))[:10]

# Ensure each user has 10 recommendations, filling in with overall top-rated songs if necessary
complete_top_recommendations = fill_with_overall_top_rated(top_user_recommendations, overall_top_rated_indices)

# Display the top 10 recommendations for each user limited to first 10 printed for clarity
limit = 10
count = 0
for user, recommendations in complete_top_recommendations.items():
    if count >= limit:
        break
    print(f"User {user}: Top 10 Recommendations - {recommendations}")
    count += 1

# Define the list of greatest hits based on some criteria (e.g., highest popularity)
greatest_hits_list = [2003, 3003, 3300, 2000, 3000, 2106, 3004, 2002, 3257, 3002]

# Function to calculate the overlap between user recommendations and the greatest hits list
def calculate_overlap(user_recommendations, hits):
    overlaps = {}
    for u, r in user_recommendations.items():
        overlap = set(r).intersection(set(hits))
        overlap_count = len(overlap)
        overlap_percentage = (overlap_count / len(hits)) * 100
        overlaps[u] = {"overlap_count": overlap_count, "overlap_percentage": overlap_percentage}
    return overlaps

# Calculate the overlap for each user
user_overlaps = calculate_overlap(complete_top_recommendations, greatest_hits_list)

# Calculate the average overlap percentage across all users
total_overlap_percentage = sum([info['overlap_percentage'] for info in user_overlaps.values()]) / len(user_overlaps)
print("Average Overlap Percentage across all users: " + str(total_overlap_percentage) + "%")

print("Extra Credit")

# Reshaping the tempo feature for model input and defining the target variable as energy
tempo = df_no_dups_ig['tempo'].values.reshape(-1, 1)
genre = df_no_dups_ig['energy'].values

x_train_11, x_test_11, y_train_11, y_test_11 = train_test_split(tempo, genre, test_size=0.15, random_state=10206068)

linreg = LinearRegression()
linreg.fit(x_train_11, y_train_11)

# Predicting energy levels on the test set using the trained model
energy_predictions = linreg.predict(x_test_11)

# Calculating the Mean Squared Error (MSE) and R-squared (R2) for model evaluation
mse_11 = mean_squared_error(y_test_11, energy_predictions)
r2_11 = r2_score(y_test_11, energy_predictions)

print("Mean Squared Error (MSE): " + str(mse_11))
print('R-squared (R2): ' +str(r2_11))

plt.scatter(x_test_11, y_test_11, color='black', label='Actual Energy')
plt.plot(x_test_11, energy_predictions, color='blue', linewidth=2, label='Predicted Energy')
plt.xlabel('Tempo')
plt.ylabel('Energy')
plt.title('Energy Prediction from Tempo')
plt.legend()
plt.show()