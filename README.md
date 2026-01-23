🌍 Language Detection App using SimpleRNN

Project Type: Machine Learning / Deep Learning  
Deployment: Streamlit Web Application  


📌 Project Overview

This project is an **end-to-end Language Detection application** built using **SimpleRNN** and deployed with **Streamlit**.
It predicts the language of a given text among **17 languages** such as English, Hindi, French, Malayalam, Spanish, Tamil, Arabic, etc.

Project Highlights:

* Multi-class text classification
* Handling class imbalance
* Complete NLP pipeline (training → evaluation → deployment)
* Simple and interpretable RNN-based model

---

🧠 Model Architecture

The model is kept simple to focus on core concepts.

Architecture:

* Embedding Layer
* SimpleRNN Layer
* Dropout Layer
* Dense Softmax Output Layer

Flow:
Embedding → SimpleRNN → Dropout → Dense (Softmax)

Why SimpleRNN?

* Easy to explain (interview-friendly)
* Captures sequential text patterns
* Lightweight and fast for deployment

---

📊 Dataset Information

Source:** Kaggle – [Language Detection Dataset](https://www.kaggle.com/datasets/basilb2s/language-detection)
Total Languages:** 17
Total Samples:** 10,267
Columns:**

  * Text (input sentence)
  * Language (target label)

⚠️ Dataset is imbalanced, so **class weights** are used.
🚀 Model Performance

| Metric        | Value     |
| ------------- | --------- |
| Test Accuracy | **95.8%** |
| Test Loss     | **0.16**  |

Improved performance due to:

* Class weighting
* Early stopping
* Proper preprocessing

---

🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Streamlit
* NumPy, Pandas, Matplotlib, Seaborn

---

📁 Project Structure


Language-Detection-RNN/
├── saved_model/
│   ├── simple_rnn_model.h5
│   └── tokenizer.pkl
├── eda.ipynb
├── prediction.ipynb
├── app.py
├── requirements.txt
└── readme.md



👤 Author

Rishabh Thakur
B.Tech  | Python | SQL | AI & ML Enthusiast

⭐ Acknowledgement

Thanks to open-source datasets and libraries that made this project possible.








🚀 Connect With Me

📧 Email: rishabhthakur5221@gmail.com
🔗 LinkedIn: www.linkedin.com/in/
🐙 GitHub: - https://github.com/rishabh-thakur-git
Thanks for checking out this project!

If this project helped you, feel free to ⭐ star the repo and share it with others learning 


