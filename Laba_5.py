import sys
from dataclasses import dataclass

import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QTableWidget,
                             QTableWidgetItem, QLabel, QGroupBox, QHBoxLayout, QPushButton, QGridLayout, QLineEdit,
                             QMessageBox, QAbstractItemView, QHeaderView, QSizePolicy)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIntValidator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure



class Equation:
    @staticmethod
    def f(x):
        return 1 / ((2 + np.cos(x)) * (3 + np.cos(x)))


class StartingPoints:
    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n
        self.h = (self.b - self.a) / (self.n - 1)
        self.degree = [2,3,4,5]

        self.x_nodes = [self.a + i * self.h for i in range (self.n)]
        self.y_nodes = [Equation.f(x) for x in self.x_nodes]

        self.x_m = [self.a + (i + 0.5) * self.h for i in range(self.n - 1)]
        self.y_m = [Equation.f(x) for x in self.x_m]
        self.x_axes = np.linspace(self.a, self.b, 100)
    def store_node_data(self):
        return NodeData(x_nodes=self.x_nodes, y_nodes=self.y_nodes, x_m=self.x_m, y_m=self.y_m, x_axes=self.x_axes)

@dataclass
class NodeData:
    x_nodes: list
    y_nodes: list
    x_m: list
    y_m: list
    x_axes: list



@dataclass
class ApproximationData:
    degree: list
    coeff: list

    y_m_polynom: list
    y_polynom: list
    polynom: str

    y_abs_ctrl: list
    abs_node: list
    max_abs_node: float
    res_node: list
    mse: float

@dataclass
class InterpolationData:
    y_lagrange_axes: list
    y_m_lagrange: list

    y_lagrange: list
    abs_error: list
    res_error: list


class ApproximationPolynom:
    def __init__(self, x_res, y_res, node_data):
        self.x_res = x_res
        self.y_res = y_res

        self.coeff = None
        self.node_data = node_data


    def poly_build(self, degree):
        n = degree + 1
        size = len(self.x_res)
        self.A = [[sum(self.x_res[k] ** (i+j) for k in range(size))
              for j in range(n)]
             for i in range(n)]
        self.B = [[sum(self.y_res[k] * (self.x_res[k] ** i)
                 for k in range(size))]
             for i in range(n)]
        return self.A, self.B

    def gauss_jordan(self):
        m = len(self.B)
        new_M = np.array([self.A[i][:] + self.B[i] for i in range(m)])

        for col in range(m):
            pivot_row = col + np.argmax(np.abs(new_M[col:, col]))
            new_M[[col, pivot_row], :] = new_M[[pivot_row, col], :]
            pivot = new_M[col, col]
            if abs(pivot) < 1e-12:
                raise ValueError('Матрица вырождена')
            new_M[col] = new_M[col] / pivot


            for row in range(m):
                if row != col:
                    factor = new_M[row, col]
                    if factor != 0:
                        new_M[row] = new_M[row] - factor * new_M[col]

        return new_M[:, -1].tolist()

    def poly_find(self, x):
        return sum(c * x**i for i, c in enumerate(self.coeff))


class BuildLagrange:
    def __init__(self, x_res, y_res, x):
        self.x_res = x_res
        self.y_res = y_res
        self.x = x

    def build_lagrange(self):
        n = len(self.x_res)
        P = 0
        for i in range(n):
            L = 1
            for j in range(n):
                if i != j:
                    L *= (self.x - self.x_res[j]) / (self.x_res[i] - self.x_res[j])
            P += self.y_res[i] * L
        return P


class Calculation:
    def __init__(self, a, b, n):
        self.points = StartingPoints(a, b, n)
        self.node_data = self.points.store_node_data()
        self.approximation_polynom = ApproximationPolynom(self.node_data.x_nodes, self.node_data.y_nodes ,self.node_data)

    def store_approximation_result(self):
        y_m = self.node_data.y_m

        y_m_polynom = [self.approximation_polynom.poly_find(x) for x in self.node_data.x_m]
        y_abs_ctrl = [abs(y_m_polynom[i] - y_m[i]) for i in range(len(y_m_polynom))]

        y_polynom = [self.approximation_polynom.poly_find(x) for x in self.node_data.x_nodes]

        abs_node = [abs(y_polynom[i] - self.node_data.y_nodes[i]) for i in range(len(y_polynom))]
        max_abs_node = max(abs_node)

        res_node = [abs_node[i] / self.node_data.y_nodes[i] * 100
                    if self.node_data.y_nodes[i] > 1e-12 else 0 for i in range(len(abs_node))]

        mse = (sum(
            (y_m_polynom[i] - y_m[i]) ** 2
            for i in range(len(y_m_polynom)))
               / len(self.node_data.x_nodes))

        polynom = ' + '.join([f'{coef:.3f}*x^{i}'
                              if i > 0 else f'{coef:.3f}'
                              for i, coef in enumerate(self.approximation_polynom.coeff)])

        return ApproximationData(
            degree=self.points.degree,
            coeff=self.approximation_polynom.coeff,
            y_m_polynom=y_m_polynom,
            y_polynom=y_polynom,

            polynom=polynom,

            y_abs_ctrl=y_abs_ctrl,
            abs_node=abs_node,
            max_abs_node=max_abs_node,
            res_node=res_node,
            mse=mse)

    def fit(self, degree):
        self.approximation_polynom.poly_build(degree)
        self.approximation_polynom.coeff = self.approximation_polynom.gauss_jordan()
        return self.store_approximation_result()

    def approximation_compute_results(self):
        polynom_data = {}

        for degree in self.points.degree:
            approximation_data = self.fit(degree)

            polynom_data[degree] = approximation_data

        best_degree = min(self.points.degree, key=lambda k: polynom_data[k].max_abs_node)

        return  polynom_data, best_degree

    def store_interpolation_data(self):
        lagrange = [
            BuildLagrange(self.node_data.x_nodes, self.node_data.y_nodes, x).build_lagrange()
            for x in self.node_data.x_m]
        y_lagrange = [lagrange[i] for i in range(len(self.node_data.x_m))]

        y_lagrange_axes = [
            BuildLagrange(self.node_data.x_nodes, self.node_data.y_nodes, x).build_lagrange()
            for x in self.node_data.x_axes]

        y_m_lagrange = [
            BuildLagrange(self.node_data.x_nodes, self.node_data.y_nodes, x).build_lagrange()
            for x in self.node_data.x_m]

        abs_error = [abs(self.node_data.y_m[i] - y_lagrange[i]) for i in range(len(self.node_data.x_m))]
        res_error = [abs_error[i] / y_lagrange[i] * 100 if abs(y_lagrange[i]) > 1e-12 else 0 for i in
                     range(len(self.node_data.x_m))]
        return InterpolationData(
            y_lagrange_axes=y_lagrange_axes,
            y_m_lagrange=y_m_lagrange,

            y_lagrange=y_lagrange,
            abs_error=abs_error,
            res_error=res_error
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.setWindowTitle('Лабораторная работа №5 | Вариант 22')
        self.resize(1450, 800)

        self.approximation_tab = ApproximationTab()
        self.interpolation_tab = InterpolationTab()


        self.main_tabs.addTab(self.approximation_tab, 'Аппроксимация')
        self.main_tabs.addTab(self.interpolation_tab, 'Интерполяция')





class ApproximationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.inner_tabs = QTabWidget()
        layout.addWidget(self.inner_tabs)
        approximation_main_tab = ApproximationGeneralTab(self)
        self.inner_tabs.addTab(approximation_main_tab, 'Сравнение всех полиномов')

    def create_inner_tabs(self, node_data, polynom_data):
        while self.inner_tabs.count() > 1:
            self.inner_tabs.removeTab(1)
        for degree in [2, 3, 4, 5]:
            tab = PolynomialTab(node_data, polynom_data[degree], degree)
            self.inner_tabs.addTab(tab, f'P{degree}')

class InterpolationTab(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.interpolation_graph = InterpolationGraphWidget()
        self.interpolation_error_table = InterpolationErrorTable()
        self.interpolation_table_info = InterpolationTableInfo()
        self.interpolation_control_panel = InterpolationControlPanel()
        self.interpolation_control_panel.update_interpolation_ui_clicked.connect(self.update_interpolation_ui_info)

        upper_layout = QHBoxLayout()
        main_layout.addLayout(upper_layout)
        upper_layout.addWidget(self.interpolation_graph)

        center_layout = QHBoxLayout()
        main_layout.addLayout(center_layout)
        center_layout.addWidget(self.interpolation_error_table)
        center_layout.addWidget(self.interpolation_table_info)

        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)
        bottom_layout.addWidget(self.interpolation_control_panel)


    def update_interpolation_ui_info(self):
        try:
            a, b, n = self.interpolation_control_panel.get_values()
            calculation = Calculation(a, b, n)
            if n < 2 or n > 10:
                raise ValueError('Введите n не меньше 2 и не более 10')

            interpolation_data = calculation.store_interpolation_data()
            node_data = calculation.node_data



            self.interpolation_graph.update_graph(node_data, interpolation_data)
            self.interpolation_error_table.update_interpolation_error_table(node_data, interpolation_data)
            self.interpolation_table_info.update_interpolation_table_info(node_data)
        except ValueError as e:
            error = QMessageBox()
            error.setText(f'Ошибка: {str(e)}')
            error.setIcon(QMessageBox.Icon.Critical)
            error.exec()

class ApproximationGeneralTab(QWidget):
    def __init__(self, approximation_tab):
        super().__init__()

        self.approximation_tab = approximation_tab


        layout = QVBoxLayout()
        self.setLayout(layout)

        upper_layout = QVBoxLayout()
        layout.addLayout(upper_layout)

        center_layout = QHBoxLayout()
        layout.addLayout(center_layout)

        lower_table_layout = QHBoxLayout()
        layout.addLayout(lower_table_layout)

        lower_layout = QHBoxLayout()
        layout.addLayout(lower_layout)


        self.matlab_file = MatlabExport()


        self.approximation_control_panel = ApproximationControlPanel()
        self.approximation_control_panel.update_approximation_ui_clicked.connect(self.update_approximation_ui_info)
        self.approximation_control_panel.matlab_export.connect(self.matlab_file.export)


        self.approximation_comparison_graph = ApproximationGraphWidget()
        upper_layout.addWidget(self.approximation_comparison_graph)

        self.node_error_table = ApproximationNodeErrorTable()

        self.check_table = ApproximationCheckTable()

        self.polynom_table = ApproximationPolynomTable()

        self.best_result_table = ApproximationBestResultsTable()



        center_layout.addWidget(self.node_error_table, 3)
        center_layout.addWidget(self.check_table, 1)

        lower_table_layout.addWidget(self.polynom_table, 2)
        lower_table_layout.addWidget(self.best_result_table, 2)


        lower_layout.addWidget(self.approximation_control_panel)





    def update_approximation_ui_info(self):
        try:
            a, b, n = self.approximation_control_panel.get_values()

            calculation = Calculation(a, b, n)
            node_data = calculation.node_data

            if n > 10 or n <2:
                raise ValueError('Введите n не меньше 2 и не более 10')
            polynom_data, best_degree = calculation.approximation_compute_results()




            self.node_error_table.update_node_table(node_data, polynom_data)
            self.check_table.update_check_table(node_data, polynom_data)
            self.polynom_table.update_polyno_table(polynom_data)
            self.best_result_table.update_best_table(polynom_data,best_degree)

            self.approximation_tab.create_inner_tabs(node_data, polynom_data)


            self.approximation_comparison_graph.update_graph(polynom_data, node_data, best_degree)


        except ValueError as e:
            error = QMessageBox()
            error.setWindowTitle('Ошибка')
            error.setText(f'Ошибка: {str(e)}')
            error.setIcon(QMessageBox.Icon.Critical)

            error.exec()


class ControlPanel(QWidget):
    update_approximation_ui_clicked = pyqtSignal()
    matlab_export = pyqtSignal()
    update_interpolation_ui_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()
        self.group = QGroupBox()
        group_layout = QGridLayout()
        self.group.setLayout(group_layout)
        self.layout.addWidget(self.group)
        self.setLayout(self.layout)

        self.a_label = QLabel('a = ')
        group_layout.addWidget(self.a_label, 0, 0)
        self.a_entry = QLineEdit('')
        group_layout.addWidget(self.a_entry, 0, 1)
        self.a_entry.insert('0')
        self.a_entry.setReadOnly(True)


        self.b_label = QLabel('b = ')
        group_layout.addWidget(self.b_label, 0, 2)
        self.b_entry = QLineEdit('')
        group_layout.addWidget(self.b_entry, 0, 3)
        self.b_entry.insert(f'{2*np.pi:.4f}')
        self.b_entry.setReadOnly(True)


        n_label = QLabel('n = ')
        group_layout.addWidget(n_label, 0, 4)
        self.n_entry = QLineEdit()
        self.n_entry.setValidator(QIntValidator(2,10))
        group_layout.addWidget(self.n_entry, 0, 5)
        self.n_entry.setPlaceholderText('Введите n (1 < n < 11)')






    def get_values(self):
        return (
            float(self.a_entry.text()),
            float(self.b_entry.text()),
            int(self.n_entry.text()))

class ApproximationControlPanel(ControlPanel):
    def __init__(self):
        super().__init__()
        button_build = QPushButton('Построить полиномы')
        button_build.clicked.connect(self.update_approximation_ui_clicked.emit)
        self.layout.addWidget(button_build)

        export_button = QPushButton('Экспорт в матлаб')
        export_button.clicked.connect(self.matlab_export.emit)
        self.layout.addWidget(export_button)


class InterpolationControlPanel(ControlPanel):
    def __init__(self):
        super().__init__()

        button = QPushButton('Решить')
        self.layout.addWidget(button)
        button.clicked.connect(self.update_interpolation_ui_clicked.emit)


class MainGraphWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        x_start = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(x_start, [Equation.f(x) for x in x_start], color='black')
        self.ax.set_title('Начальный полином')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.grid(True)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(250)


class ApproximationGraphWidget(MainGraphWidget):
    def __init__(self):
        super().__init__()
    
    def update_graph(self, polynom_data, node_data, best_degree):
        self.ax.clear()
        self.plot_function(node_data)
        self.scatter_nodes(node_data)
        self.plot_polynom(node_data, polynom_data, best_degree)
        self.decorations()


    def plot_function(self, node_data):
        x_axes = node_data.x_axes
        self.ax.plot(x_axes, [Equation.f(x) for x in x_axes], label='f(x)', color='#3e0b9c')


    def scatter_nodes(self, node_data):
        self.ax.scatter(node_data.x_nodes, node_data.y_nodes, color='#3e0b9c', label='Узлы')

    def plot_polynom(self, node_data, polynom_data, best_degree):
        colors = ['#5854e8', '#54dce8', '#27964e', '#912020']

        for i, degree in enumerate([2, 3, 4, 5]):
            lbl = f'P{degree}'
            self.ax.plot(node_data.x_nodes, polynom_data[degree].y_polynom,
                         color=colors[i], label=lbl, ls='-' if degree == best_degree else '--')

    def decorations(self):
        self.ax.legend(loc='upper right', fontsize=7)
        self.ax.grid(True)
        self.ax.axhline(0, color='black', linestyle='-')
        self.ax.axvline(0, color='black', linestyle='-')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Сравнение полиномов степени 2, 3, 4, 5')

        self.canvas.draw()


class InterpolationGraphWidget(MainGraphWidget):
    def __init__(self):
        super().__init__()
    def update_graph(self, node_data, interpolation_data):
        self.ax.clear()
        self.plot_function(node_data)
        self.scatter_plot(node_data)
        self.plot_lagrange(node_data, interpolation_data)
        self.decorations()

    def plot_function(self, node_data):
        self.ax.plot(
            node_data.x_axes,
            [Equation.f(float(x)) for x in node_data.x_axes],
            ls='-',
            color='#1d1024',
            label='f(x)'
        )

    def scatter_plot(self, node_data):
        self.ax.scatter(
            node_data.x_nodes,
            node_data.y_nodes,
            c='#1d1024', label='Узлы', s=25)

    def plot_lagrange(self, node_data, interpolation_data):
        self.ax.plot(
            node_data.x_axes,
            interpolation_data.y_lagrange_axes,
            c='#6423db', label='P(x)', ls='--')

        self.ax.scatter(
            node_data.x_m,
            interpolation_data.y_m_lagrange,
            marker='v', c='#6423db',
            label='Промежут. точки', s=50, edgecolor='k')

    def decorations(self):
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Интерполяция Лагранжа')
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.5)

        self.canvas.draw()


class TableWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)


class ApproximationNodeErrorTable(TableWidget):
    def __init__(self):
        super().__init__()

        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(['xi', 'P2 абс.', 'P3 абс.',
                                                   'P4 абс.', 'P5 абс.', 'P2 отн.',
                                                   'P3 отн.', 'P4 отн.', 'P5 отн.'])
        self.table.setFixedHeight(200)


    def update_node_table(self, node_data, polynom_data):
        count = len(node_data.x_nodes)
        self.table.setRowCount(0)

        for row in range(count):
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0,
                               QTableWidgetItem(str(f'{node_data.x_nodes[row]:.5f}')))

            for i, degree in enumerate([2, 3, 4, 5]):
                self.table.setItem(row, i + 1,
                                   QTableWidgetItem(str(f'{polynom_data[degree].abs_node[row]:.5f}')))

                self.table.setItem(row, i + 5,
                                   QTableWidgetItem(str(f'{polynom_data[degree].res_node[row]:.5f}%')))
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ApproximationCheckTable(TableWidget):
    def __init__(self):
        super().__init__()

        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['xi', 'P2', 'P3', 'P4', 'P5'])
        self.table.setFixedHeight(200)



    def update_check_table(self, node_data, polynom_data):
        self.table.setRowCount(0)
        count = len(node_data.x_m)
        for row in range(count):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(f'{node_data.x_m[row]:.5f}')))

            for i, degree in enumerate([2, 3, 4, 5]):
                self.table.setItem(row, i + 1,
                                         QTableWidgetItem(str(f'{polynom_data[degree].y_abs_ctrl[row]:.5f}')))
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ApproximationPolynomTable(TableWidget):
    def __init__(self):
        super().__init__()

        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['степень', 'мнк', 'полимер'])
        self.table.setFixedHeight(140)



    def update_polyno_table(self, polynom_data):


        self.table.setRowCount(0)
        
        for i, degree in enumerate([2, 3, 4, 5]):
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(str(f'P{degree}')))
            self.table.setItem(row, 1, QTableWidgetItem(str(f'{polynom_data[degree].mse:.5f}')))
            self.table.setItem(row, 2, QTableWidgetItem(str(polynom_data[degree].polynom)))
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ApproximationBestResultsTable(TableWidget):
    def __init__(self):
        super().__init__()

        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Лучший полином', "мнк", "степень"])
        self.table.setRowCount(1)
        self.table.setFixedHeight(50)



    def update_best_table(self, polynom_data, best_degree):

        self.table.setRowCount(0)


        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(str(polynom_data[best_degree].polynom)))
        self.table.setItem(row, 1, QTableWidgetItem(str(f'{polynom_data[best_degree].mse:.5f}')))
        self.table.setItem(row, 2, QTableWidgetItem(str(best_degree)))


class InterpolationErrorTable(TableWidget):
    def __init__(self):
        super().__init__()

        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["xⱼ", "f(xⱼ)", "P(xⱼ)", "Абс. погрешность", "Отн. погрешность"])
        self.table.setFixedHeight(300)


    def update_interpolation_error_table(self, node_data, interpolation_data):
        self.table.setRowCount(0)

        count = len(node_data.x_m)
        for row in range(count):
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(str(f'{node_data.x_m[row]:.5f}')))
            self.table.setItem(row, 1, QTableWidgetItem(str(f'{node_data.y_m[row]:.5f}')))
            self.table.setItem(row, 2, QTableWidgetItem(str(f'{interpolation_data.y_lagrange[row]:.5f}')))
            self.table.setItem(row, 3, QTableWidgetItem(str(f'{interpolation_data.abs_error[row]:.5f}')))
            self.table.setItem(row, 4, QTableWidgetItem(str(f'{interpolation_data.res_error[row]:.5f}%')))


class InterpolationTableInfo(TableWidget):
    def __init__(self):
        super().__init__()

        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['xi', 'f(xi)'])
        self.table.setFixedHeight(300)


    def update_interpolation_table_info(self, node_data):
        self.table.setRowCount(0)
        count = len(node_data.x_nodes)

        for row in range(count):
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(str(f'{node_data.x_nodes[row]:.5f}')))
            self.table.setItem(row, 1, QTableWidgetItem(str(f'{node_data.y_nodes[row]:.5f}')))


class PolynomialTab(QWidget):
    def __init__(self, node_data, polynom_data, degree):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        upper_layout = QVBoxLayout()
        layout.addLayout(upper_layout)

        lower_layout = QVBoxLayout()
        layout.addLayout(lower_layout)


        self.graph = GraphWidget()
        self.table = TableError(degree)

        self.graph.set_data(node_data, polynom_data)
        self.table.set_data(polynom_data)

        upper_layout.addWidget(self.graph)
        lower_layout.addWidget(self.table)


class GraphWidget(MainGraphWidget):
    def __init__(self):
        super().__init__()

    def set_data(self, node_data, polynom_data):
        self.redraw(node_data, polynom_data)

    def redraw(self, node_data, polynom_data):
        self.ax.clear()
        self.ax.clear()
        self.ax.plot(node_data.x_axes, [Equation.f(x) for x in node_data.x_axes], label='f(x)', color='#3e0b9c')
        self.ax.scatter(node_data.x_nodes, node_data.y_nodes, color='#3e0b9c')



        self.ax.plot(node_data.x_nodes, polynom_data.y_polynom, label=f'P{polynom_data.degree}')
        self.ax.legend(loc='upper right', fontsize=7)
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.canvas.draw()


class TableError(TableWidget):
    def __init__(self, degree):
        super().__init__()

        self.table.setColumnCount(2)

        layout = QVBoxLayout()
        group = QGroupBox(f'P{degree}')
        group_layout = QVBoxLayout(group)

        group_layout.addWidget(self.table)
        self.layout.addWidget(group)


        self.table.setHorizontalHeaderLabels(['Абс. погрешность' ,'Отн. погрешность',])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(400)

    def set_data(self, polynom_data):
        self.table.setRowCount(0)
        for i in range(len(polynom_data.abs_node)):
            self.data_append(polynom_data.abs_node[i], polynom_data.res_node[i])

    def data_append(self, abs_node, res_node):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(f'{abs_node:.5f}')))
        self.table.setItem(row, 1, QTableWidgetItem(str(f'{res_node:.5f}%')))


class MatlabExport:
    def __init__(self):


        self.script = """clc;
clear;
close all;

func = @(x) 1 ./ ((2 + cos(x)) .* (3 + cos(x)));

a = 0;
b = 2*pi;
n = 10;

x = linspace(a, b, n + 1);
y = func(x);

h = (b - a) / n;
x_m = a + h/2 : h : b - h/2;
y_m = func(x_m);

x_pl = linspace(a, b, 250);
y_pl = func(x_pl);

degree = [2, 3, 4, 5];
colors = {'r', 'b', 'g', 'k'};

figure(1);

plot(x_pl, y_pl, 'Color', 'm', 'LineWidth', 2, 'DisplayName', 'f(x)');

hold on;

plot(x, y, 'ro', 'MarkerSize', 7, 'MarkerFaceColor', 'm', 'DisplayName', 'Узловые точки');

figure(2);

for j = 1:length(degree)

    deg = degree(j);
    p = polyfit(x, y, deg);

    fprintf('Коэффициенты полинома P%d:\\n', deg);

    for k = 1:length(p)
        power = deg - (k - 1);

        fprintf('a%d = %.10f\\n', power, p(k));
    end

    y_ap = polyval(p, x);
    y_pl_ap = polyval(p, x_pl);
    y_m_ap = polyval(p, x_m);

    abs_err = abs(y - y_ap);

    rel_err = abs_err ./ abs(y) * 100;

    mse = mean((y - y_ap).^2);

    figure(1);

    plot(x_pl, y_pl_ap, ...
        'Color', colors{j}, ...
        'LineWidth', 1.5, ...
        'DisplayName', sprintf('P%d', deg));

    figure(2);

    subplot(2,2,j);

    plot(x_pl, y_pl, 'm', 'LineWidth', 2);

    hold on;

    plot(x_pl, y_pl_ap, 'r--', 'LineWidth', 1.5);

    plot(x, y, 'ko', 'MarkerFaceColor', 'k');

    title(sprintf('Полином степени %d', deg));

    xlabel('x');
    ylabel('y');

    legend('f(x)', sprintf('P%d', deg), 'Узлы');

    grid on;

    fprintf('\\nСтепень полинома = %d\\n', deg);
    fprintf('MSE = %.8f\\n\\n', mse);

    fprintf('%-5s %-12s %-14s %-16s %-16s\\n', 'i', 'x_i', 'y_i', 'Абс. погр.', 'Отн. погр.');

    for i = 1:length(x)

        fprintf('%-5d %-12.6f %-14.6f %-16.6f %-16.6f\\n', i, x(i), y(i), abs_err(i), rel_err(i));

    end

    fprintf('\\nМаксимальная ошибка = %.8f\\n', max(abs_err));

    fprintf('========================================\\n\\n');

end


figure(1);

title('Сравнение полиномов');

xlabel('x');
ylabel('y');

legend('Location', 'best');

grid on;"""

    def export(self):
        with open('Vini_Lab_5.m', 'w', encoding='utf-8') as f:

            f.write(self.script)
        download = QMessageBox()
        download.setText('Файл был успешно сохранен')
        download.setIcon(QMessageBox.Icon.Information)

        download.exec()



app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec())
