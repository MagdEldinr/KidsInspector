import PIL.Image as Img
import PIL.ImageTk as ImageTk
from tkinter import*
from tkinter.ttk import*
import tkinter as tk
import tkinter.messagebox
import random
import math as m
from test import predict
from os import listdir
from os.path import isfile, join
import cv2
targetImage = []
temp_window = []


root = Tk()
root.iconbitmap(default='icon.ico')
#root.resizable(0,0)
class Inspector(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.last_selected_img = None
        self.labels = []
        self.results = []
        self.imageNames =[]

    def initUI(self):
        self.master.title("Kids Inspector")
        self.pack(fill=BOTH, expand=1)
        self.centerMainWindow()


        Style().configure("TFrame", background="#7B9897")

        def onObjectClick_right(event):
            answer = tk.messagebox.askquestion("Congrats", "Play Again")
            if answer == "yes":
                questionButton.config(state=DISABLED)
                viewframe.delete(ALL)
                browseButtonHandler()
            else:
                root.destroy()

        def onObjectClick_wrong(event):
            tk.messagebox.showinfo("Worng , Try Again !")

        labelIntro = Label(self, text="Welcome To Kids Inspector, Choose An Image!", font=("Calibri", 18))
        labelIntro.place(x=20, y=10)

        frame = tk.Frame(self, width=530, height=485, bg="white", highlightbackground="black",
                         highlightcolor="black", highlightthickness=2)
        frame.place(x=20, y=50)

        #viewframe = tk.Frame(frame, width=696, height=396, bg="white")
        viewframe = Canvas(frame, width=482, height=473, bg="white")
        viewframe.place(x=20, y=0)

        def browseButtonHandler():
            viewframe.delete(ALL)
            questionButton.config(state = DISABLED)
            for child in viewframe.winfo_children():
                child.destroy()

            quizButton.configure(state=DISABLED)
            browseButton.configure(state=DISABLED)
            browseWindow = Toplevel(self)
            for window in temp_window:
                window.destroy()
            temp_window.clear()

            def onBrowseWinClose():
                browseButton.configure(state=NORMAL)
                temp_window.append(browseWindow)
                browseWindow.withdraw()

            # browseWindow.protocol('WM_DELETE_WINDOW', onBrowseWinClose)
            # browseWindow.overrideredirect(1)
            browseWindow.title("Browse Images")
            self.centerGenericWindow(browseWindow, 550, 600)
            scrollbar = Scrollbar(browseWindow)
            scrollbar.pack(side=RIGHT, fill=Y)

            text = Text(browseWindow, yscrollcommand=scrollbar.set, height=100)
            text.pack()

            imgLabels = []
            self.imageNames = [f for f in listdir("images") if isfile(join("images", f))]
            for i,img in enumerate(self.imageNames):
                path = "./images/" + img
                imgLabels.append(self.createImgLabel(path, text, (448, 448)))
                text.window_create(END, window=imgLabels[i-1])
                text.insert(END, '\n\n')
            scrollbar.config(command=text.yview)

            def labelClick(event):
                for label in imgLabels:
                    label.configure(background="black")
                event.widget.configure(background="red")
                self.last_selected_img = imgLabels.index(event.widget)
                targetImage.clear()
                targetImage.append(event.widget["image"])
                #print(event.widget["image"])
                selectButton.configure(state=NORMAL)

            for label in imgLabels:
                label.bind("<Button-1>", labelClick)

            def selectButtonHandler():
                onBrowseWinClose()
                #self.viewTargetImage(viewframe, targetImage[0])
                viewframe.create_image(241, 234, image=targetImage[0])
                quizButton.configure(state=NORMAL)

            text.insert(END, '\t\t\t')
            selectButton = tk.Button(text, text="Open Image", state=DISABLED, command=selectButtonHandler,
                                     width=15, font=("Calibri", 14))
            text.window_create(END, window=selectButton)
            text.insert(END, '\n')
            text.config(state=DISABLED)

        def quizButtonHandler():
            # Function Body
            questionButton.config(state="normal")
            path = './images/' + self.imageNames[self.last_selected_img]
            print(path)
            img = cv2.imread(path)
            self.labels , self.results = predict(img)
            print(self.labels)
            random.seed(9001)
            object_index = random.randint(0, len(self.labels)-1)
            question.config(text="Click on " + self.labels[object_index])
            for i,box in enumerate(self.results):
                x2, y2, h2, w2 = box[0], box[1], box[2], box[3]
                click_area1 = viewframe.create_rectangle(x2 - w2, y2 - h2, x2 + 1.1*w2, y2 + 1.1*h2, width=3,

                                                     outline="maroon", stipple="gray25", fill="white")
                if i == object_index:
                    viewframe.tag_bind(click_area1, '<ButtonPress-1>', onObjectClick_right)
                else:
                    viewframe.tag_bind(click_area1, '<ButtonPress-1>', onObjectClick_wrong)

        def ask_new_question():
            viewframe.create_image(241, 234, image=targetImage[0])
            object_index = random.randint(0, len(self.labels)-1)
            question.config(text="Click on " + self.labels[object_index])
            for i, box in enumerate(self.results):
                x2, y2, h2, w2 = box[0], box[1], box[2], box[3]
                print(box)
                click_area1 = viewframe.create_rectangle(x2 - w2, y2 - h2, x2 + w2, y2 + h2, width=3,

                                                         outline="maroon", stipple="gray25", fill="white")
                if i == object_index:
                    viewframe.tag_bind(click_area1, '<ButtonPress-1>', onObjectClick_right)
                else:
                    viewframe.tag_bind(click_area1, '<ButtonPress-1>', onObjectClick_wrong)



        def questionButtonHandler():
            # Function Body
            viewframe.delete(ALL)
            ask_new_question()


        frameButtons = tk.Frame(self, width=250, height=240, bg="white", highlightbackground="black",
                            highlightcolor="black", highlightthickness=2)
        frameButtons.place(x=580, y=50)

        controlLabel = Label(self, text="Control Panel", font=("Calibri", 18))
        controlLabel.place(x=640, y=10)

        browseButton = tk.Button(frameButtons, text="Browse Images", command=browseButtonHandler, width=20,
                                 font=("Calibri", 14))
        browseButton.place(x=18, y=20)

        quizButton = tk.Button(frameButtons, text="Start Quiz", command=quizButtonHandler, width=20,
                                 font=("Calibri", 14))
        quizButton.place(x=18, y=70)
        quizButton.configure(state=DISABLED)

        questionButton = tk.Button(frameButtons, text="Change Question", command=questionButtonHandler, width=20,
                               font=("Calibri", 14))
        questionButton.place(x=18, y=120)
        questionButton.configure(state=DISABLED)

        closeButton = tk.Button(frameButtons, text="Close", command=quit, width=20,
                                   font=("Calibri", 14))
        closeButton.place(x=18, y=170)


        questionLabel = Label(self, text="Answer Quiz Questions:", font=("Calibri", 18))
        questionLabel.place(x=20, y=540)

        questionFrame = tk.Frame(self, width=530, height=100, bg="white", highlightbackground="black",
                                highlightcolor="black", highlightthickness=2)
        questionFrame.place(x=20, y=580)

        question_text = tk.StringVar()
        question_text.set("Here Goes A Question?")
        question = Label(questionFrame, text = "Here Goes A Question?", font=("Calibri", 14), background="white")
        question.place(x=5, y=5)

        instructionFrame = tk.Frame(self, width=250, height=240, bg="white", highlightbackground="black",
                                highlightcolor="black", highlightthickness=2)
        instructionFrame.place(x=580, y=300)

        innerFrame = tk.Frame(instructionFrame, width=246, height=236, bg="white")
        innerFrame.place(x=123, y=118, anchor="center")
        innerFrame.grid_propagate(False)

        instructions = Text(innerFrame, wrap=CHAR)
        instructions.place(x=0, y=0)
        instructions.insert(END, '1. Click Browse Images button \n')
        instructions.insert(END, 'to select and open an image.\n\n')
        instructions.insert(END, '2. Click Start Quiz button \n')
        instructions.insert(END, 'to answer questions.\n\n')
        instructions.insert(END, '3. Click on the target object \n')
        instructions.insert(END, 'on the image to answer.\n\n')
        instructions.insert(END, '4. Click on Change Question \n')
        instructions.insert(END, 'to get a new question.\n\n')
        instructions.insert(END, '5. Click on Close button \n')
        instructions.insert(END, 'to close the program.\n\n')
        instructions.configure(state=DISABLED)

    def centerMainWindow(self):
        w = 860
        h = 710

        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()

        x = (sw - w) / 2
        y = (sh - h) / 2
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def centerGenericWindow(self, window, w, h):
        sw = window.master.winfo_screenwidth()
        sh = window.master.winfo_screenheight()
        x = (sw - w) / 2
        y = (sh - h) / 2
        window.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def createImgLabel(self, path, parent, size):
        img_choice = Img.open(path)
        img_choice = img_choice.resize(size, Img.ANTIALIAS)
        img_choice_tk = ImageTk.PhotoImage(img_choice)
        imgLabel = Label(parent, image=img_choice_tk, background="black")
        imgLabel.image = img_choice_tk
        return imgLabel

    def viewTargetImage(self, viewframe, image):
        target = image

        imgLabel = Label(viewframe, image=image[0], background="black")
        imgLabel.image = image[0]
        imgLabel.place(x=348, y=198, anchor="center")
def main():
    app = Inspector()
    photo = ImageTk.PhotoImage(file="logo.jpg")
    label_photo = Label(root, image = photo)
    #label_photo.place(x=580, y=550)
    root.mainloop()


if __name__ == '__main__':
    main()