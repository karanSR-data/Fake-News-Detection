# predict.py
from predict_pipeline import predict_news

if __name__ == "__main__":
    while True:
        text = input("Enter news text (or 'q' to quit): ")

        if text.lower() == 'q':
            break

        result = predict_news(text)
        print("Prediction:", result)

# “Bhai, ek simple test karna hai model ka.”
# → Debugging
# → Terminal testing
# → Quick check