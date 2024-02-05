from utils import read_csv
import tkinter as tk
import matplotlib.pyplot as plt
from perceptron_model import Perceptron
from tkinter.filedialog import askopenfilename
from tkinter import ttk


class MenuUI:
    def __init__(self):
        self.__master = tk.Tk()
        self.__master.title("Perceptron")
        self.__master.geometry("400x300")
        self.__file_csv_path = None

        self.__create_widgets()

    def start(self):
        self.__master.mainloop()

    def __create_widgets(self):
        frame = tk.Frame(self.__master)
        frame.pack(fill=tk.BOTH, expand=True)

        general_frame = tk.LabelFrame(frame, text="Datos generales")
        general_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        self.__learning_rate_label = tk.Label(general_frame, text="Tasa de aprendizaje")
        self.__learning_rate_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.__learning_rate_entry = tk.Entry(general_frame)
        self.__learning_rate_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.__permissible_error_label = tk.Label(
            general_frame, text="Error permisible"
        )
        self.__permissible_error_label.grid(
            row=1, column=0, padx=10, pady=10, sticky="w"
        )

        self.__permissible_error_entry = tk.Entry(general_frame)
        self.__permissible_error_entry.grid(
            row=1, column=1, padx=10, pady=10, sticky="ew"
        )

        self.__iterations_label = tk.Label(general_frame, text="Iteraciones")
        self.__iterations_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.__iterations_entry = tk.Entry(general_frame)
        self.__iterations_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.__file_label = tk.Label(general_frame, text="Archivo")
        self.__file_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.__file_button = tk.Button(
            general_frame, text="Seleccionar archivo", command=self.__open_file
        )
        self.__file_button.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

        self.__train_button = tk.Button(
            general_frame, text="Entrenar", command=self.__train
        )
        self.__train_button.grid(
            row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

    def __open_file(self):
        file = askopenfilename(
            title="Seleccionar archivo", filetypes=[("CSV Files", "*.csv")]
        )

        if file:
            file_name = file.split("/")[-1]
            self.__file_label.config(text=f"Archivo: {file_name}")
            self.__file_csv_path = file

    def __train(self):
        learning_rate = float(self.__learning_rate_entry.get())
        permissible_error = float(self.__permissible_error_entry.get())
        iterations = int(self.__iterations_entry.get())
        file_path = self.__file_csv_path

        model = Perceptron(learning_rate, permissible_error, iterations)

        data = read_csv(file_path)

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        model.fit(X, y)

        # Plot the error rule
        error_rule = model.get_errors_history()
        plt.figure(1)

        plt.plot(error_rule, label="Error")
        plt.title("Normal del |E|")
        plt.xlabel("# Iteraciones")
        plt.ylabel("Norma del error")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.tight_layout()

        # Plot the weights history
        weights_history = model.get_weights_history()
        plt.figure(2)

        for i in range(len(weights_history[0])):
            weights = [weights[i] for weights in weights_history]
            plt.plot(weights, label=f"W{i}")

        plt.title("Evolución de los pesos")
        plt.xlabel("# Iteraciones")
        plt.ylabel("Valor del peso")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.tight_layout()

        results_win = tk.Toplevel(self.__master)
        results_win.title("Resultados")
        results_win.geometry("700x600")

        result_frame = tk.Frame(results_win)
        result_frame.pack(fill=tk.BOTH, expand=True)

        err_rule_table_label = tk.Label(
            result_frame, text="Tabla de evolución del error (|e|)"
        )
        err_rule_table_label.pack()

        err_rule_table = ttk.Treeview(
            result_frame, columns=("Iteración", "Norma del error"), show="headings"
        )

        err_rule_table.heading("Iteración", text="Iteración")
        err_rule_table.heading("Norma del error", text="Norma del error")

        for i, error in enumerate(error_rule):
            err_rule_table.insert("", i, values=(i, error))

        err_rule_table.pack()

        weights_table_label = tk.Label(
            result_frame, text="Tabla de evolución de los pesos (W)"
        )
        weights_table_label.pack()

        num_weights = len(weights_history[0])
        weight_col_names = ["Iteración"] + [f"W{i}" for i in range(num_weights)]

        weights_table = ttk.Treeview(
            result_frame, columns=(weight_col_names), show="headings"
        )

        weights_table.heading("Iteración", text="Iteración")
        for i in range(num_weights):
            weights_table.heading(f"W{i}", text=f"W{i}")

        for i, weights in enumerate(weights_history):
            weights_table.insert("", i, values=[i] + list(weights))

        weights_table.pack()

        initial_final_weights_label = tk.Label(
            result_frame, text=f"Configuración de pesos inicial: {weights_history[0]}"
        )
        initial_final_weights_label.pack()

        initial_final_weights_label = tk.Label(
            result_frame, text=f"Configuración de pesos final: {weights_history[-1]}"
        )
        initial_final_weights_label.pack()

        learning_rate_label = tk.Label(
            result_frame, text=f"Tasa de aprendizaje utilizada: {learning_rate}"
        )
        learning_rate_label.pack()

        permissible_error_label = tk.Label(
            result_frame, text=f"Error permisible: {permissible_error}"
        )
        permissible_error_label.pack()

        iterations_label = tk.Label(
            result_frame, text=f"Cantidad de iteraciones: {iterations}"
        )
        iterations_label.pack()

        plt.show()
