# import necessary libraries modules
import pickle
import os.path
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
import PIL
import PIL.Image
import PIL.ImageDraw
import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class DrawingClassifier:
    background_image = None

    # default constructor
    def __init__(self):
        self.background_label = None
        self.background_photo = None
        self.welcome_root = None
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None
        self.brush_width = 15
        self.background_image = Image.open("welcome_background.jpg")
        self.setup_welcome_screen()

    # ask shape name that to be trained
    def classes_prompt(self):
        message = Tk()
        message.withdraw()
        message.iconbitmap("icon.ico")
        self.proj_name = simpledialog.askstring(
            "Project Name", "Please enter your project name down below!", parent=message
        )
        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class1 = data["c1"]
            self.class2 = data["c2"]
            self.class3 = data["c3"]
            self.class1_counter = data["c1c"]
            self.class2_counter = data["c2c"]
            self.class3_counter = data["c3c"]
            self.clf = data["clf"]
            self.proj_name = data["pname"]
        else:
            self.class1 = simpledialog.askstring(
                "Class 1", "What is the first class called?", parent=message
            )
            self.class2 = simpledialog.askstring(
                "Class 2", "What is the second class called?", parent=message
            )
            self.class3 = simpledialog.askstring(
                "Class 3", "What is the third class called?", parent=message
            )

            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1

            self.clf = LinearSVC()

            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    # basic GUI
    def init_gui(self):
        width = 500
        height = 500
        white = (255, 255, 255)

        self.root = Tk()
        self.root.iconbitmap("icon.ico")
        self.root.title(f"Artificial Intelligence - {self.proj_name}")

        self.canvas = Canvas(
            self.root, width=width - 10, height=height - 10, bg="white"
        )
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (width, height), white)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        bm_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        bp_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W + E)

        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)

        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)

        save_everything_btn = Button(
            btn_frame, text="Save Everything", command=self.save_everything
        )
        save_everything_btn.grid(row=3, column=2, sticky=W + E)

        self.status_label = Label(
            btn_frame, text=f"Current Model: {type(self.clf).__name__}"
        )
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    # Add the rest of the methods with their existing functionality

    def setup_welcome_screen(self):
        self.welcome_root = Tk()
        self.welcome_root.title("Artificial Intelligence App")

        self.welcome_root.iconbitmap("icon.ico")
        self.welcome_root.title("Welcome to the Artificial Intelligence App")
        self.welcome_root.geometry("600x400")  # Adjust the dimensions as per your image

        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = Label(self.welcome_root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        welcome_label = Label(
            self.welcome_root,
            text="Welcome to the Artificial Intelligence App",
            font=("Arial", 20),
            bg="black",
            fg="white",
        )
        welcome_label.pack(pady=20)

        start_button = Button(
            self.welcome_root,
            text="Start Drawing",
            command=self.start_drawing,
            bg="black",
            fg="white",
            font=("Arial", 16),
            relief=RAISED,
            padx=10,
            pady=5,
        )
        start_button.pack(pady=20)
        self.welcome_root.mainloop()
        self.welcome_root.destroy()

    def start_drawing(self):
        self.classes_prompt()
        self.init_gui()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill="black", width=self.brush_width
        )
        self.draw.rectangle(
            [x1, y2, x2 + self.brush_width, y2 + self.brush_width],
            fill="black",
            width=self.brush_width,
        )

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel(
            "Quit?", "Do you want to save your work?", parent=self.root
        )
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.ANTIALIAS)

        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.proj_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1

        self.clear()

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.proj_name}/{self.class3}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(
            self.class1_counter - 1 + self.class2_counter - 1 + self.class3_counter - 1,
            2500,
        )

        self.clf.fit(img_list, class_list)
        tkinter.messagebox.showinfo(
            "Artificial Intelligence App",
            "Model successfully trained!",
            parent=self.root,
        )

    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.ANTIALIAS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo(
                "Artificial Intelligence App",
                f"The drawing is probably a {self.class1}",
                parent=self.root,
            )
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo(
                "Artificial Intelligence App",
                f"The drawing is probably a {self.class2}",
                parent=self.root,
            )
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo(
                "Artificial Intelligence",
                f"The drawing is probably a {self.class3}",
                parent=self.root,
            )

    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()

        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo(
            "Artificial Intelligence", "Model successfully saved!", parent=self.root,
        )

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo(
            "Artificial Intelligence App",
            "Model successfully loaded!",
            parent=self.root,
        )

    def save_everything(self):
        data = {
            "c1": self.class1,
            "c2": self.class2,
            "c3": self.class3,
            "c1c": self.class1_counter,
            "c2c": self.class2_counter,
            "c3c": self.class3_counter,
            "clf": self.clf,
            "pname": self.proj_name,
        }
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo(
            "Artificial Intelligence App",
            "Project successfully saved!",
            parent=self.root,
        )

    def start_drawing_classifier(self, welcome_root):
        welcome_root.destroy()
        self.init_gui()


DrawingClassifier()
