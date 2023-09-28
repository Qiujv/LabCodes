"""GUI program for modifying crosstalk matrix."""


import ast
import math
import re
import tkinter as tk

import matplotlib as mpl
import numpy as np


class MatrixEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Matrix Editor")
        self.create_widgets()

        self.cmap = mpl.colormaps.get_cmap('RdBu_r')
        self.norm = mpl.colors.Normalize(-0.1, 0.1)

    def create_widgets(self):
        txt_label = tk.Label(self.master, text="Enter the matrix:")
        txt_label.pack(padx=5, pady=5)

        self.txt_entry = tk.Text(self.master, height=5, width=50)
        self.txt_entry.insert(tk.END, "[[1,-0.03,0],[0,1,0.05],[0,0.01,1]]")
        self.txt_entry.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        buttons_frame = tk.Frame(self.master)
        buttons_frame.pack(padx=5, pady=5)

        display_botton = tk.Button(buttons_frame, text="⇓ Display", command=self.txt_to_matrix)
        display_botton.pack(side=tk.LEFT, padx=5, pady=5)

        output_botton = tk.Button(buttons_frame, text="⇑ To text", command=self.matrix_to_txt)
        output_botton.pack(side=tk.RIGHT, padx=5, pady=5)

        label_frame = tk.Frame(self.master)
        label_frame.pack(padx=5, pady=5, fill=tk.X, expand=True)

        label_label = tk.Label(label_frame, text="Space:")
        label_label.pack(side=tk.LEFT)

        self.labels = tk.StringVar(value="['Q1', 'Q2', 'Q3']")
        self.label_entry = tk.Entry(label_frame, width=40, textvariable=self.labels)
        self.label_entry.bind('<Key-Return>', lambda event: self.refresh_matrix())
        self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.matrix_frame = tk.Frame(self.master)
        self.matrix_frame.pack(padx=5, pady=5)
        # Make the window width scale with the matrix frame
        self.master.columnconfigure(0, weight=1)

    def txt_to_matrix(self):
        matrix = np.array(ast.literal_eval(self.txt_entry.get("1.0", tk.END)))
        self.set_matrix(matrix)

    def matrix_to_txt(self):
        matrix = self.get_matrix()
        txt = np.array2string(matrix, separator=',', suppress_small=True)
        txt = re.sub(r'(?<=\d)\.(?=\D)', '.0', txt)
        self.txt_entry.delete("1.0", tk.END)
        self.txt_entry.insert(tk.END, txt)

    def set_matrix(self, matrix: np.matrix):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        # clear all context menu widget created before, if any.
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Menu):
                widget.destroy()

        ndim = matrix.shape[0]

        labels = ast.literal_eval(self.labels.get())
        if len(labels) < ndim:
            labels = labels + list(range(len(labels), ndim))
        self.labels.set(str(labels))

        largest_width = 4
        for i in range(ndim):
            for j in range(ndim):
                val = matrix[i,j]
                bg_color = self.cmap(self.norm(val))
                txt_color = 'black' if mpl.colors.rgb_to_hsv(bg_color[:3])[2] > 0.5 else 'white'
                bg_color = mpl.colors.to_hex(bg_color)
                element_entry = tk.Entry(self.matrix_frame, width=4, background=bg_color, foreground=txt_color)
                element_entry.insert(0, val)
                element_entry.bind('<Key-Return>', self.update_color)
                element_entry.bind("<Button-3>", self.context_menu)
                element_entry.grid(row=i+1, column=j+1, padx=0, pady=0)
                create_tooltip(element_entry, f'{labels[i]} by {labels[j]}')
                if len(str(val)) > largest_width:
                    largest_width = len(str(val))

        for lb in labels:
            if len(str(lb)) > largest_width:
                largest_width = len(str(lb))

        for i in range(ndim):
            row_label = tk.Label(self.matrix_frame, text=labels[i])
            row_label.grid(row=i+1, column=0, padx=0, pady=0)

        for j in range(ndim):
            col_label = tk.Label(self.matrix_frame, text=labels[j])
            col_label.grid(row=0, column=j+1, padx=0, pady=0)

        for widget in self.matrix_frame.winfo_children():
            widget.config(width=largest_width)

    def get_matrix(self) -> np.matrix:
        mat = [float(widget.get()) for widget in self.matrix_frame.winfo_children() 
               if isinstance(widget, tk.Entry)]
        ndim = math.isqrt(len(mat))
        return np.matrix(mat).reshape(ndim, -1)
    
    def refresh_matrix(self):
        matrix = self.get_matrix()
        self.set_matrix(matrix)

    def insert_matrix(self, idx: int):
        labels = ast.literal_eval(self.labels.get())
        labels.insert(idx, '_')
        self.labels.set(str(labels))

        matrix = self.get_matrix()
        matrix = np.insert(matrix, idx, 0, axis=0)
        matrix = np.insert(matrix, idx, 0, axis=1)
        self.set_matrix(matrix)

    def delete_dim(self, idx: int):
        labels = ast.literal_eval(self.labels.get())
        labels.pop(idx)
        self.labels.set(str(labels))

        matrix = self.get_matrix()
        matrix = np.delete(matrix, idx, axis=0)
        matrix = np.delete(matrix, idx, axis=1)
        self.set_matrix(matrix)

    def context_menu(self, event: tk.Event) -> None:
        element_entry = event.widget
        row_idx = element_entry.grid_info()['row'] - 1
        col_idx = element_entry.grid_info()['column'] - 1
        menu = tk.Menu(self.master, tearoff=0)
        menu.add_command(label="Add column before", command=lambda: self.insert_matrix(col_idx))
        menu.add_command(label="Add column after", command=lambda: self.insert_matrix(col_idx+1))
        menu.add_command(label="Add row above", command=lambda: self.insert_matrix(row_idx))
        menu.add_command(label="Add row below", command=lambda: self.insert_matrix(row_idx+1))
        menu.add_command(label="Delete column", command=lambda: self.delete_dim(col_idx))
        menu.add_command(label="Delete row", command=lambda: self.delete_dim(row_idx))
        menu.post(event.x_root, event.y_root)

    def update_color(self, event: tk.Event):
        element_entry = event.widget
        val = float(element_entry.get())
        bg_color = self.cmap(self.norm(val))
        txt_color = 'black' if mpl.colors.rgb_to_hsv(bg_color[:3])[2] > 0.5 else 'white'
        bg_color = mpl.colors.to_hex(bg_color)
        element_entry.config(background=bg_color, foreground=txt_color)

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None

    def show_tip(self):
        if self.tip_window or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + cy
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

def create_tooltip(widget, text):
    tool_tip = ToolTip(widget, text)
    widget.bind('<Enter>', lambda event: tool_tip.show_tip())
    widget.bind('<Leave>', lambda event: tool_tip.hide_tip())


if __name__ == "__main__":
    root = tk.Tk()
    app = MatrixEditor(master=root)
    app.mainloop()


# A larger matrix for test
[[ 1.     ,  0.013  , -0.0255 ,  0.0119 ,  0.     ,  0.     ,
    0.     ,  0.     ,  0.     ,  0.     ],
[ 0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,
    0.     ,  0.     ,  0.     ,  0.     ],
[-0.0081 , -0.016  ,  1.     ,  0.0251 ,  0.     ,  0.     ,
    0.     ,  0.     ,  0.     ,  0.     ],
[ 0.     ,  0.     ,  0.0569 ,  1.     ,  0.     ,  0.     ,
    0.     ,  0.     ,  0.     ,  0.     ],
[ 0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.017  ,
    -0.0169 ,  0.0094 ,  0.     ,  0.     ],
[ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,
    0.     ,  0.     ,  0.     ,  0.     ],
[ 0.     ,  0.     ,  0.     ,  0.     , -0.0058 , -0.0114 ,
    1.     ,  0.01965,  0.014  ,  0.0042 ],
[ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,
    0.0382 ,  1.     ,  0.     ,  0.     ],
[ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,
    0.     ,  0.     ,  1.     ,  0.     ],
[ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,
    0.0093 ,  0.     ,  0.024  ,  1.     ]]
['Q1', 'C12', 'Q2', 'G2', 'Q4', 'C45', 'Q5', 'G5', 'C56', 'Q6']
