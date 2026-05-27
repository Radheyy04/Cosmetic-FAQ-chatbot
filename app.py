from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)
# FAQ Dataset
faq_questions = [
    "Which skincare products are best for oily skin?",
    "Do you sell products for dry skin?",
    "Are your cosmetic products cruelty free?",
    "Do you offer organic beauty products?",
    "Which face wash is suitable for sensitive skin?",
    "How can I choose the correct foundation shade?",
    "Do your products contain harmful chemicals?",
    "Are your lipsticks waterproof?",
    "Which sunscreen is best for daily use?",
    "Do you offer products for acne treatment?",
    "Can I return opened cosmetic products?",
    "How long does delivery take?",
    "Do you provide cash on delivery?",
    "How can I track my order?",
    "Are there discounts on beauty products?",
    "Do you sell products for men?",
    "Which shampoo is best for hair fall?",
    "Do you have anti aging skincare products?",
    "How do I know if a product suits my skin?",
    "Do your products have expiry dates?",
    "Can I exchange damaged products?",
    "Do you offer makeup kits for beginners?",
    "Which moisturizer is best for winter?",
    "Are your products dermatologist tested?",
    "Do you provide international shipping?",
    "Can I cancel my order after placing it?",
    "Which serum is best for glowing skin?",
    "Do you sell vegan cosmetic products?",
    "How should I store beauty products?",
    "Do you offer free shipping?"
]

faq_answers = [
    "We offer oil free skincare products specially designed for oily skin.",
    "Yes, we provide moisturizers and skincare products for dry skin.",
    "Yes, many of our cosmetic products are cruelty free.",
    "Yes, organic beauty and skincare products are available.",
    "Gentle and fragrance free face washes are recommended for sensitive skin.",
    "You can use our shade matching guide to select the right foundation shade.",
    "Most of our products are free from parabens and harmful chemicals.",
    "Yes, many of our lipsticks are waterproof and long lasting.",
    "SPF 30 or higher sunscreen is recommended for daily use.",
    "Yes, we provide face washes, creams, and serums for acne treatment.",
    "Opened cosmetic products cannot be returned for hygiene reasons.",
    "Delivery usually takes 3 to 5 business days.",
    "Yes, cash on delivery is available in selected locations.",
    "You can track your order from the My Orders section.",
    "Yes, discounts and seasonal offers are available on selected products.",
    "Yes, we offer grooming and skincare products for men.",
    "Strengthening shampoos with biotin are recommended for hair fall control.",
    "Yes, we offer anti aging creams, serums, and skincare kits.",
    "You can check product descriptions and skin type recommendations before buying.",
    "Yes, all cosmetic products include manufacturing and expiry dates.",
    "Yes, damaged products can be exchanged within 7 days of delivery.",
    "Yes, beginner friendly makeup kits are available.",
    "Hydrating moisturizers with shea butter are ideal for winter.",
    "Yes, many products are dermatologist tested and skin safe.",
    "Yes, international shipping is available for selected countries.",
    "Orders can be cancelled before they are shipped.",
    "Vitamin C serum is highly recommended for glowing skin.",
    "Yes, vegan beauty and skincare products are available.",
    "Beauty products should be stored in a cool and dry place.",
    "Yes, free shipping is available on orders above 999 rupees."
]

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_words = []
    for word in tokens:
        if word not in stop_words and word not in string.punctuation:
            filtered_words.append(word)
    return " ".join(filtered_words)

processed_questions = [preprocess(q) for q in faq_questions]
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]
    processed_input = preprocess(user_message)
    user_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.2:
        return jsonify({
            "reply": "Sorry, I could not understand your question."
        })
    response = faq_answers[best_match]
    return jsonify({
        "reply": response
    })

if __name__ == "__main__":
    app.run(debug=True)