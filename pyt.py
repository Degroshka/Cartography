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
    [1, 0, 1, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1]
])

# Функция для расчета кумулятивной вероятности
def calculate_cumulative_probabilities(probabilities):
    q = [0] * len(probabilities)
    q[0] = 0
    for i in range(1, len(probabilities)):
        q[i] = q[i - 1] + probabilities[i - 1]
    return q

# Кодирование Шеннона с добавлением проверочных битов
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
        
        # Добавление четного или нечетного бита
        parity_bit = '1' if code_word.count('1') % 2 == 0 else '0'
        code_dict[symbol] = code_word + parity_bit  # Код + проверочный бит
    return code_dict
# Кодирование последовательности
def encode_with_parity(sequence, code_dict):
    encoded_sequence = ''
    for symbol in sequence:
        if symbol not in code_dict:
            messagebox.showerror("Error", f"Символ '{symbol}' не найден в кодировочном словаре.")
            return
        encoded_sequence += code_dict[symbol]  # Используем код уже с проверочным битом
    return encoded_sequence
# Декодирование последовательности с проверкой на четность
def decode_with_parity(encoded_sequence, code_dict):
    reverse_dict = {v: k for k, v in code_dict.items()}
    decoded_sequence = []
    errors = []
    code_length = len(next(iter(code_dict.values())))  # Длина кодового слова уже с проверочным битом
    i = 0
    while i < len(encoded_sequence):
        code = encoded_sequence[i:i + code_length]
        if len(code) < code_length:
            messagebox.showerror("Error", "Закодированная последовательность имеет неверную длину.")
            return "", []
        parity_bit = code[-1]  # Последний бит — проверочный
        code_without_parity = code[:-1]  # Код без проверочного бита
        expected_parity = '1' if code_without_parity.count('1') % 2 == 0 else '0'
        
        if expected_parity != parity_bit:
            errors.append(f"Ошибка на позиции {i // code_length + 1}: {code} (ожидался паритет {expected_parity})")
        decoded_sequence.append(reverse_dict.get(code, '?'))
        i += code_length

    return ''.join(decoded_sequence), errors

# Кодирование Хэмминга
def hamming_coding(symbols):
    code_dict = {}
    for i, symbol in enumerate(symbols):
        binary_rep = format(i, 'b').zfill(4)  # Представляем индекс в двоичном формате (4 бита)
        data_bits = np.array([int(bit) for bit in binary_rep])
        encoded_bits = np.dot(data_bits, G_matrix) % 2  # Кодируем данные через G-матрицу
        code_dict[symbol] = ''.join(map(str, encoded_bits))  # Генерируем код
    return code_dict

# Декодирование Хэмминга

def hamming_decoding(encoded_sequence, code_dict):
    reverse_dict = {v[:4]: k for k, v in code_dict.items()}  # Information bits -> symbol
    valid_codes = set(code_dict.values())  # Все валидные кодовые слова
    decoded_sequence = ""
    errors = []
    code_length = 7  # Длина кодового слова (7,4)
    i = 0
    while i < len(encoded_sequence):
        code = encoded_sequence[i:i + code_length]

        if len(code) != code_length:
            errors.append(f"Неверный код: {code}")
            i += code_length
            continue

        code_array = np.array([int(bit) for bit in code])
        
        # Вычисление синдрома
        syndrome = np.dot(H_matrix, code_array) % 2
        syndrome_decimal = int("".join(map(str, syndrome)), 2)

        if syndrome_decimal != 0:
            # Ошибка в бите syndrome_decimal (нумерация с 1, справа налево)
            errors.append(f"Ошибка в бите {syndrome_decimal} для кода: {code}")
            if 1 <= syndrome_decimal <= code_length:
                # Исправляем ошибку, меняем бит в позиции syndrome_decimal
                error_index = code_length - syndrome_decimal  # Преобразуем синдром в индекс с конца
                code_array[error_index] = 1 - code_array[error_index]  # Инвертируем бит
                corrected_code = ''.join(map(str, code_array))
                errors.append(f"Исправленный код: {corrected_code}")
            else:
                errors.append(f"Невозможно исправить код: {code}")
                corrected_code = code 
        else:
            corrected_code = code  # Ошибок нет

        if corrected_code in valid_codes:
            data_bits = corrected_code[:4]  # Извлекаем информационные биты
            decoded_symbol = reverse_dict.get(data_bits, '?')
            decoded_sequence += decoded_symbol
        else:
            errors.append(f"Неизвестный символ для: {corrected_code}")

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
        self.title("Кодирование Шеннона и Хэмминга")
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
        self.calculate_probs_button = tk.Button(self, text="Рассчитать коды", command=self.calculate_probabilities)
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
        self.errors_text = tk.Text(self, height=6, width=40)
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
        if not symbols:
            messagebox.showerror("Error", "Введите символы.")
            return
        self.symbols = symbols
        self.probabilities = [1 / len(symbols)] * len(symbols)
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
        if self.coding_type.get() == "shannon":
            encoded_sequence = encode_with_parity(sequence, self.code_dict)
        else:
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
        if self.coding_type.get() == "shannon":
            decoded_sequence, errors = decode_with_parity(encoded_sequence, self.code_dict)
        else:
            decoded_sequence, errors = hamming_decoding(encoded_sequence, self.code_dict)
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
            messagebox.showinfo("Success", "Последовательность успешно загружена")

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
            messagebox.showinfo("Success", "Закодированная последовательность успешно сохранена")
if __name__ == "__main__":
    app = ShannonApp()
    app.mainloop()
