import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
# from softwaretech_capstone import FunctionGeneratePrediction
from softwaretech_capstone import FunctionPredictResult

class DiamondPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Diamond Price Prediction')
        self.data = pd.read_csv('diamonds.csv')
        self.sliders = []

        self.X = self.data.drop('price', axis=1).values
        self.y = self.data['price'].values

        ## self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # self.model = svm.SVR()
        # self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i, column=2)
            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i, column=1)
            self.sliders.append((slider, current_val_label))

        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)

    def predict_price(self):
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Price', f'The predicted house price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    print("Main running sucessfully")
    app = DiamondPricePredictionApp(root)
    root.mainloop()