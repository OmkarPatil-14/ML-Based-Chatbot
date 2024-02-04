from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import tkinter as tk
from tkinter import font
from dataset_good import dataset
from PIL import Image, ImageTk



class BankingChatbot:
    def __init__(self):
        # Create a machine learning model
        self.ml_model = make_pipeline(CountVectorizer(), MultinomialNB())

    def train_ml_model(self, dataset):
        # Assuming the dataset is a list of tuples (query, response)
        queries, responses = zip(*dataset)
        self.ml_model.fit(queries, responses)

    def predict_ml_response_with_confidence(self, query):
        # Use the machine learning model to predict the response and get probability estimates
        probs = self.ml_model.predict_proba([query])[0]

        # Find the index of the class with the highest probability
        predicted_class_index = np.argmax(probs)

        # Get the corresponding response
        predicted_response = self.ml_model.classes_[predicted_class_index]

        # Get the confidence (probability) of the predicted response
        confidence = probs[predicted_class_index]

        return predicted_response, confidence

    def respond(self, message):
        # Use the machine learning model to predict the response with confidence
        response, confidence = self.predict_ml_response_with_confidence(message)

        # if confidence < 0.0500 :
        #     print("Sorry, I didn't understand your query, please elaborate clearly...")
        #     run_bot()
        #
        # else:
        return f'{response} with confidence: {confidence:.4f}'


# Example usage with a dataset of queries and responses
banking_chatbot = BankingChatbot()
dataset = dataset()
# Assuming you have a dataset variable containing your dataset
banking_chatbot.train_ml_model(dataset)


def display_open():
    display.config(state=tk.NORMAL)

def display_close():
    display.config(state=tk.DISABLED)

def update_display():
    input_text = user_input.get()
    if input_text == "X" or input_text == "x":
        display_open()
        display.insert(tk.END,"Chatbot closed")
        display_close()

    elif "X" not in input_text or "x" not in input_text:
        display_open()
        display.insert(tk.END,"You : " + input_text + "\n\n")
        response = banking_chatbot.respond(input_text)
        display.insert(tk.END,"Chatbot : " + response + "\n\n")
        display_close()
        user_input.delete(0, tk.END)  # Clear the user input box

# Create the main window
root = tk.Tk()
root.title("Chatbot Application")

window_size = 550
root.geometry(f"{850}x{700}")

# Set background color to white
root.configure(bg="white")

# Load and resize the logo image
logo_image = Image.open("logo.png")  # Replace "logo.png" with your actual image file
logo_image = logo_image.resize((60, 40))  # Resize the image as needed
logo_image = ImageTk.PhotoImage(logo_image)

# Create a heading label with the logo
heading_frame = tk.Frame(root, bg="white")
heading_frame.pack(pady=10)

logo_label = tk.Label(heading_frame, image=logo_image, bg="white")
logo_label.grid(row=0, column=0,padx=10)

bold_font = font.Font(weight="bold", size=30)

# Create a label and an entry box for Display
heading_label = tk.Label(heading_frame, text="HSBC Bank Chatbot ", font=bold_font,bg="white",fg="black")
heading_label.grid(row=0, column=1, pady=15)


# Create a display box
prewritten_text = "\nHi, this is HSBC chatbot.\n\nChatbot : Hello there! Enter your query in the input box or press X to exit \n\n"
display = tk.Text(root, height=30, width=110, font=("Helvetica", 12, "bold"), state=tk.DISABLED, bg="white", fg="black")
display.config(state=tk.NORMAL)
display.insert(tk.END, prewritten_text)
display.config(state=tk.DISABLED)
display.pack(padx=20)


# Create a label and an entry box for user input
query_label = tk.Label(root, text="Enter your Query or Press X to exit : ", bg="white", fg="black")
query_label.pack(pady=12)

# Create an entry box for user input
user_input = tk.Entry(root, width=85, bg="white",fg="black")
user_input.pack(pady=10, ipady=10)

# Create a "GO" button
go_button = tk.Button(root, text="GO", width=8, command=update_display)
go_button.pack()

# Run the Tkinter event loop
root.mainloop()
