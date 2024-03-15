import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Sample email data (you can replace this with your dataset)
emails = [
    ("Hey, let's catch up this weekend!", "Personal"),
    ("Invitation to join my LinkedIn network.", "Social"),
    ("Buy one get one free offer inside!", "Others"),
    ("Congratulations! You won a trip to Hawaii.", "Others"),
    ("Reminder: Tomorrow's party at 7 PM.", "Social"),
    ("Monthly newsletter: Updates and promotions.", "Others"),
    ("Meeting agenda for next week's conference.", "Others"),
    ("Happy birthday! Have a fantastic day!", "Personal")
]

# Preprocess text
stop_words = set(stopwords.words("english"))
preprocessed_emails = []
for email, label in emails:
    words = word_tokenize(email.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    preprocessed_email = " ".join(filtered_words)
    preprocessed_emails.append((preprocessed_email, label))

# Split data into training and testing sets
random.shuffle(preprocessed_emails)
train_data, test_data = train_test_split(preprocessed_emails, test_size=0.2)

# Train the classifier
classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
classifier.fit([email for email, label in train_data], [label for email, label in train_data])

# Evaluate the classifier
accuracy = classifier.score([email for email, label in test_data], [label for email, label in test_data])
print("Accuracy:", accuracy)

# Test the classifier with new emails
new_emails = [
    "Hey, how have you been?",
    "Join me for a party tonight!",
    "Limited time offer: 50% discount!",
    "Reminder: Doctor's appointment tomorrow.",
    "Congratulations on your promotion!"
]

for email in new_emails:
    words = word_tokenize(email.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    preprocessed_email = " ".join(filtered_words)
    prediction = classifier.predict([preprocessed_email])[0]
    print(f"Email: {email} - Predicted category: {prediction}")
