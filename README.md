# 📩 Email/SMS Spam Classifier

A machine learning-based web application that classifies SMS or email text as **Spam** or **Not Spam (Ham)** using a voting ensemble model. Built with **Python**, **Scikit-learn**, and **Streamlit** for the frontend interface.

---

## 🚀 Features

- Text preprocessing (lowercase, stopword removal, stemming)
- TF-IDF vectorization (top 3000 features)
- Ensemble classification using:
  - 🧠 Multinomial Naive Bayes
  - 🔍 Support Vector Classifier (SVC)
  - 🌲 Extra Trees Classifier
- Voting classifier and stacking model support
- Streamlit frontend for easy user interaction
- Confidence score (%) for predictions

---


---

## 🛠 Tech Stack

| Tool/Library     | Purpose                      |
|------------------|------------------------------|
| Python           | Core programming language    |
| Pandas, NumPy    | Data manipulation            |
| Matplotlib, Seaborn | Data visualization        |
| Scikit-learn     | ML model building + metrics  |
| NLTK             | Text preprocessing           |
| Streamlit        | Frontend web UI              |
| Pycharm          | Development IDE              |
| Pickle           | Model serialization          |

---

## 🧪 Model Evaluation

- **Accuracy**: ~98%
- **Precision**: ~97%
- **Test Size**: 20%
- **Dataset**: `spam.csv` (approx. 5,500 messages)

---

## 🧰 Installation & Usage

### 1️⃣ Clone the Repository

```bash

```
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```bash

pip install -r requirements.txt

streamlit run app.py
```
```bash

.
├── app.py               # Streamlit app
├── model.pkl            # Saved Stacking Classifier model
├── vectorizer.pkl       # TF-IDF vectorizer
├── spam.csv             # Dataset
├── requirements.txt     # Dependencies
└── README.md
```
```bash

📌 Example Spam Messages
"🎁 Congratulations! You've won a free iPhone. Claim now at http://bit.ly/xyz"

"🚨 Your bank account is suspended. Click here to verify your identity immediately!"

"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"

👤 Author
Gautam Jyala
BTech | Developer | ML Enthusiast

---

### ✅ requirements.txt (create a separate file)

Create a file named `requirements.txt` and paste this:




