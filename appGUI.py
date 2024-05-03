import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from apifunctions import FunctionGeneratePrediction
import json

class DiamondPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Diamond Price Prediction')
        self.data = pd.read_csv('diamonds.csv')
        self.sliders = []

        self.X = self.data.drop('price', axis=1).values
        self.y = self.data['price'].values

        # model is trained separately so we don't have to wait for it to re-train each time we use the GUI
        # run trainModel.py to retrain the model and generate the pickle files

        self.create_widgets()

    def create_widgets(self):
        widgetCols = ["carat", "x", "y", "z"]
        for i, column in enumerate(self.data.columns):
            print("column=" + column)
            if column in widgetCols:
                label = tk.Label(self.master, text=column + ': ')
                label.grid(row=i, column=0)
                current_val_label = tk.Label(self.master, text='0.0')
                current_val_label.grid(row=i, column=2)

                slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                                command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
                slider.grid(row=i, column=1)
                self.sliders.append((slider, current_val_label))

        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns), columnspan=3)

    def predict_price(self):
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        prediction = FunctionGeneratePrediction(inp_carat=inputs[0],inp_x= inputs[1],inp_y= inputs[2],inp_z= inputs[3])
        results = json.loads(prediction)
        price = results['Prediction']['0']
        messagebox.showinfo('Predicted Price', f'The predicted diamond price is ${price:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    print("Main running sucessfully")
    app = DiamondPricePredictionApp(root)
    root.mainloop()