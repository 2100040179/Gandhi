# Import necessary libraries
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens dataset
data = Dataset.load_builtin('ml-100k')

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm for collaborative filtering
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Predict ratings for the testset
predictions = algo.test(testset)

# Calculate and print the RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Function to get top n recommendations for a specific user
def get_top_n_recommendations(predictions, user_id, n=10):
    # Filter predictions for the specific user
    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    
    # Sort the predictions by estimated rating in descending order
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top n recommendations
    top_n = user_predictions[:n]
    return top_n

# Get top 10 recommendations for a specific user
user_id = '196'
top_n_recommendations = get_top_n_recommendations(predictions, user_id, n=10)

# Print the top 10 recommendations for the user
print(f"Top 10 recommendations for user {user_id}:")
for pred in top_n_recommendations:
    print(f"Item ID: {pred.iid}, Estimated Rating: {pred.est:.2f}")
