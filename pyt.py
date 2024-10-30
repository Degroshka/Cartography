import tkinter as tk
from tkinter import messagebox, filedialog
import math
import numpy as np

# Матрица кодирования и проверочная матрица для кодов Хэмминга
G_matrix = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1]
])
H_matrix = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
])

# Функция для расчета кумулятивной вероятности
def calculate_cumulative_probabilities(probabilities):
    q = [0] * len(probabilities)
    for i in range(1, len(probabilities)):
        q[i] = q[i - 1] + probabilities[i - 1]
    return q

# Кодирование Шеннона
def shannon_coding(symbols, probabilities):
    code_dict = {}
    cumulative_probs = calculate_cumulative_probabilities(probabilities)
    for i, symbol in enumerate(symbols):
        prob = probabilities[i]
        cum_prob = cumulative_probs[i]
        code_length = math.ceil(-math.log2(prob))
        code_word = ""
        fractional_part = cum_prob
        for _ in range(code_length):
            fractional_part *= 2
            if fractional_part >= 1:
                code_word += "1"
                fractional_part -= 1
            else:
                code_word += "0"
        code_dict[symbol] = code_word
    return code_dict

# Декодировка Шеннона
def shannon_decoding(encoded_sequence, code_dict):
    reverse_dict = {v: k for k, v in code_dict.items()}
    decoded_sequence = ""
    current_code = ""
    for bit in encoded_sequence:
        current_code += bit
        if current_code in reverse_dict:
            decoded_sequence += reverse_dict[current_code]
            current_code = ""
    return decoded_sequence

# Переводит строку бинарного кода в массив numpy
def binary_str_to_array(binary_str):
    return np.array([int(bit) for bit in binary_str])

# Переводит массив numpy в строку бинарного кода
def array_to_binary_str(array):
    return ''.join(str(bit) for bit in array)

# Кодирование Хэмминга
def hamming_coding(symbols):
    code_dict = {}
    for i, symbol in enumerate(symbols):
        binary_rep = format(i, 'b').zfill(4)  # бинарное представление символа
        data_bits = binary_str_to_array(binary_rep)
        encoded_bits = np.dot(data_bits, G_matrix) % 2
        code_dict[symbol] = array_to_binary_str(encoded_bits)
    return code_dict

# Декодирование Хэмминга с исправлением ошибок
def hamming_decoding(encoded_sequence, code_dict):
    reverse_dict = {v: k for k, v in code_dict.items()}
    decoded_sequence = ""
    errors = []
    code_length = 7

    i = 0
    while i < len(encoded_sequence):
        code = encoded_sequence[i:i + code_length]
        code_array = binary_str_to_array(code)

        # Рассчитываем синдром для обнаружения ошибок
        syndrome = np.dot(H_matrix, code_array) % 2
        syndrome_decimal = int("".join(map(str, syndrome)), 2)

        if syndrome_decimal != 0:
            error_message = f"Обнаружена ошибка в бите {syndrome_decimal} для кода: {code}"
            errors.append(error_message)

            if 1 <= syndrome_decimal <= code_length:
                code_array[syndrome_decimal - 1] = (code_array[syndrome_decimal - 1] + 1) % 2
                corrected_code = array_to_binary_str(code_array)
                error_message += f". Исправлен код: {corrected_code}"
                errors.append(error_message)

            if corrected_code not in reverse_dict:
                reverse_dict[corrected_code] = reverse_dict.get(code, '?')
        else:
            corrected_code = code  # нет ошибок

        decoded_symbol = reverse_dict.get(corrected_code, '?')
        if decoded_symbol == '?':
            errors.append(f"Не удалось декодировать символ для кода: {corrected_code}")
        decoded_sequence += decoded_symbol
        i += code_length
    return decoded_sequence, errors

# Вычисление метрик кода
def calculate_code_metrics(code_dict):
    codes = list(code_dict.values())
    code_lengths = set(len(code) for code in codes)
    if len(code_lengths) > 1:
        messagebox.showerror("Error", "Все коды должны быть одинаковой длины для вычисления метрик.")
        return None, None, None, None
    try:
        d_min = float('inf')
        for i, c1 in enumerate(codes):
            for c2 in codes[i + 1:]:
                distance = sum(x != y for x, y in zip(c1, c2))
                if distance < d_min:
                    d_min = distance
    except ValueError:
        messagebox.showerror("Error", "Ошибка при вычислении расстояния Хэмминга.")
        return None, None, None, None
    if d_min <= 1:
        messagebox.showerror("Error", "Границы нельзя рассчитать с расстоянием меньшим чем 2.")
        return d_min, None, None, None
    n = len(codes[0])
    M = len(codes)
    hamming_bound = 2 ** n / sum(math.comb(n, k) for k in range((d_min - 1) // 2 + 1))
    plotkin_bound = (1 / (1 - (d_min / n))) if d_min / n < 0.5 else None
    varshamov_gilbert_bound = 2 ** n / sum(math.comb(n, k) for k in range(d_min - 1))
    return d_min, hamming_bound, plotkin_bound, varshamov_gilbert_bound

# Класс для интерфейса tkinter
class ShannonApp(tk.Tk):
    # Инициализация и настройка интерфейса
    def __init__(self):
        super().__init__()
        self.title("Кодирование с проверкой на чётность")
        self.geometry("800x600")
        # Поля ввода
        self.symbols_label = tk.Label(self, text="Символы алфавита")
        self.symbols_label.place(x=50, y=40)
        self.symbols_entry = tk.Entry(self, width=50)
        self.symbols_entry.place(x=50, y=60)
        self.sequence_label = tk.Label(self, text="Последовательность для кодирования")
        self.sequence_label.place(x=50, y=120)
        self.sequence_entry = tk.Entry(self, width=50)
        self.sequence_entry.place(x=50, y=140)
        # Радиокнопки для выбора типа кодировки
        self.coding_type = tk.StringVar(value="shannon")
        self.shannon_radio = tk.Radiobutton(self, text="Кодировка Шеннона", variable=self.coding_type, value="shannon")
        self.hamming_radio = tk.Radiobutton(self, text="Кодировка Хэмминга", variable=self.coding_type, value="hamming")
        self.shannon_radio.place(x=50, y=10)
        self.hamming_radio.place(x=200, y=10)
        # Кнопки действий
        self.calculate_probs_button = tk.Button(self, text="Рассчитать вероятности и коды", command=self.calculate_probabilities)
        self.calculate_probs_button.place(x=50, y=90)
        self.encode_button = tk.Button(self, text="Закодировать с проверочными битами", command=self.encode_sequence)
        self.encode_button.place(x=50, y=165)
        self.decode_button = tk.Button(self, text="Декодировать и проверить на ошибки", command=self.decode_sequence)
        self.decode_button.place(x=50, y=245)
        self.load_button = tk.Button(self, text="Загрузить последовательность из файла", command=self.load_sequence)
        self.load_button.place(x=400, y=140)
        self.save_button = tk.Button(self, text="Сохранить закодированную последовательность", command=self.save_encoded_sequence)
        self.save_button.place(x=400, y=215)  
        # Кнопка для вычисления метрик
        self.calculate_metrics_button = tk.Button(self, text="Вычислить", command=self.calculate_code_metrics_after_encoding)
        self.calculate_metrics_button.place(x=190, y=390)
        # Поля для отображения результатов
        self.code_label = tk.Label(self, text="Закодированная последовательность")
        self.code_label.place(x=50, y=195)
        self.code_text = tk.Entry(self, width=50)
        self.code_text.place(x=50, y=215)
        self.decode_label = tk.Label(self, text="Раскодированная последовательность")
        self.decode_label.place(x=50, y=290)
        self.decode_text = tk.Entry(self, width=50)
        self.decode_text.place(x=50, y=310)
        self.errors_label = tk.Label(self, text="Ошибки")
        self.errors_label.place(x=400, y=400)
        self.errors_text = tk.Text(self, height=6, width=35)
        self.errors_text.place(x=400, y=420)
        self.code_dict_text = tk.Text(self, height=6, width=25)
        self.code_dict_text.place(x=500, y=20)
        # Поле для метрик кода
        self.metrics_label = tk.Label(self, text="Кодовые метрики")
        self.metrics_label.place(x=50, y=400)
        self.metrics_text = tk.Text(self, height=6, width=35)
        self.metrics_text.place(x=50, y=420)
        # Инициализация данных
        self.symbols = []
        self.probabilities = []
        self.code_dict = {}

    # Рассчитать вероятности и создать коды
    def calculate_probabilities(self):
        symbols = self.symbols_entry.get().strip().split()
        n = len(symbols)
        if n == 0:
            messagebox.showerror("Error", "Введите символы.")
            return
        self.symbols = symbols
        self.probabilities = [1 / n] * n
        self.code_dict = shannon_coding(self.symbols, self.probabilities) if self.coding_type.get() == "shannon" else hamming_coding(self.symbols)
        self.update_code_dict_display()

    # Кодировать последовательность
    def encode_sequence(self):
        sequence = self.sequence_entry.get().strip()
        if not sequence:
            messagebox.showerror("Error", "Введите последовательность.")
            return
        if not self.code_dict:
            messagebox.showerror("Error", "Сначала рассчитайте вероятности и коды.")
            return
        encoded_sequence = ''.join(self.code_dict.get(symbol, '?') for symbol in sequence)
        self.code_text.delete(0, tk.END)
        self.code_text.insert(0, encoded_sequence)

    # Декодировать последовательность
    def decode_sequence(self):
        encoded_sequence = self.code_text.get().strip()
        if not encoded_sequence:
            messagebox.showerror("Error", "Введите закодированную последовательность.")
            return
        if not self.code_dict:
            messagebox.showerror("Error", "Сначала рассчитайте вероятности и коды.")
            return
        decoded_sequence, errors = (shannon_decoding(encoded_sequence, self.code_dict), []) if self.coding_type.get() == "shannon" else hamming_decoding(encoded_sequence, self.code_dict)
        self.decode_text.delete(0, tk.END)
        self.decode_text.insert(0, decoded_sequence)
        self.errors_text.delete(1.0, tk.END)
        self.errors_text.insert(tk.END, "\n".join(errors) if errors else "Ошибок нет")

    # Обновить отображение словаря кодов
    def update_code_dict_display(self):
        self.code_dict_text.delete(1.0, tk.END)
        for symbol, code in self.code_dict.items():
            self.code_dict_text.insert(tk.END, f"{symbol}: {code}\n")

    # Вычислить метрики кода
    def calculate_code_metrics_after_encoding(self):
        if not self.code_dict:
            messagebox.showerror("Error", "Коды ещё не созданы.")
            return
        d_min, hamming_bound, plotkin_bound, varshamov_gilbert_bound = calculate_code_metrics(self.code_dict)
        self.metrics_text.delete(1.0, tk.END)
        if d_min is not None:
            self.metrics_text.insert(tk.END, f"Кодовое расстояние (d_min): {d_min}\n")
        if hamming_bound is not None:
            self.metrics_text.insert(tk.END, f"Граница Хэмминга: {hamming_bound:.2f}\n")
        if plotkin_bound is not None:
            self.metrics_text.insert(tk.END, f"Граница Плоткина: {plotkin_bound:.2f}\n")
        if varshamov_gilbert_bound is not None:
            self.metrics_text.insert(tk.END, f"Граница Варшамова-Гильберта: {varshamov_gilbert_bound:.2f}\n")

    # Загрузить последовательность из файла
    def load_sequence(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                sequence = file.read().strip()
            self.sequence_entry.delete(0, tk.END)
            self.sequence_entry.insert(0, sequence)

    # Сохранить закодированную последовательность в файл
    def save_encoded_sequence(self):
        encoded_sequence = self.code_text.get().strip()
        if not encoded_sequence:
            messagebox.showerror("Error", "Нет закодированной последовательности для сохранения.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(encoded_sequence)
            messagebox.showinfo("Success", f"Закодированная последовательность сохранена в {file_path}")

if __name__ == "__main__":
    app = ShannonApp()
    app.mainloop()
