import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

training_data = pd.DataFrame({
    'email': [
        'Hey, this is a legitimate email.',
        'Congratulations! You have won a prize.',
        'Claim your exclusive offer now!',
        'Important meeting tomorrow.',
        'Please review the attached document.',
        'URGENT: Your account has been compromised.',
        "You've won a lottery",
        "You've received a direct deposit of $30,000",
        'Get 150 free spins now - Join today!',
        "Dear Michael, I hope this email finds you well.",
        "Hi Customer, your Apple ID will be disabled because of some violated policies."
    ],
    'label': ['not spam', 'spam', 'spam', 'not spam', 'not spam', 'spam', 'spam', 'spam', 'spam', 'not spam', 'spam']
})

# Split the dataset into features (emails) and labels (spam or not spam)
emails = training_data['email']
labels = training_data['label']

# Convert text data into numerical features using the Bag-of-Words model
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(emails)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Evaluate the classifier on the testing set
accuracy = naive_bayes.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Example predictions on new data
new_emails = [
    'Limited time offer: 50% off on selected items!',
    'Reminder: Team meeting at 2 PM today.',
    'Get a free trial for our new product.',
    'Hey Joe, want to get a cup of coffee?',
    "Dear Mary, please attend the meeting tomrrow."
]

new_features = vectorizer.transform(new_emails)
predictions = naive_bayes.predict(new_features)

# print("Predictions:")
# for email, prediction in zip(new_emails, predictions):
#     print(f"{email} - {prediction}")