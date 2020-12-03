#Author: Jordan Bennett
#Note: Simple ui to use Ai based coronavirus2019/covid19 diagnosis tool

import covid_diagnosis


####Drag and Drop imports
# This must be imported before PIL imports below, to avoid Image Opening error
from TkinterDnD2 import *

try:
    from Tkinter import *
except ImportError:
    from tkinter import *
####

from tkinter import Frame, Tk, BOTH, Label, Menu, filedialog, messagebox, Text
from PIL import Image, ImageTk

import os
import codecs

screenWidth = "560"
screenHeight = "660"
windowTitle = "Smart/Ai Coronavirus 2019 (Covid19) Diagnosis Tool"

import cv2
from random import randint


class Window(Frame):
    _PRIOR_IMAGE = None
    DIAGNOSIS_RESULT = ""
    DIAGNOSIS_RESULT_FIELD = None

    # Jordan_note: Added to facilitate output window data

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        load = Image.open("covid19_ai_diagnoser_default__.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=(int(screenWidth) / 2) - load.width / 2, y=((int(screenHeight) / 2)) - load.height / 2 - 80)
        self._PRIOR_IMAGE = img  # setup prior image instance
        self.DIAGNOSIS_RESULT_FIELD = Text(self, width=int(screenWidth), height=13)
        self.DIAGNOSIS_RESULT_FIELD.pack()
        self.DIAGNOSIS_RESULT_FIELD.place(x=0, y=int(screenHeight) - 200)

    def addDiagnosisResult(self, value):
        self.DIAGNOSIS_RESULT_FIELD.delete("1.0", "end")  # clear diagnostic result text element
        self.DIAGNOSIS_RESULT = ""  # clear diagnostic result string variable
        self.DIAGNOSIS_RESULT_FIELD.insert(1.0, value)  # add new value


root = TkinterDnD.Tk()  # Root changed from Tk() to TkinterDnD.Tk() to facilitate drag/drop
app = Window(root)

############
# screen window size, window title
root.wm_title(windowTitle)
root.geometry(screenWidth + "x" + screenHeight)

############
# menu bar
menubar = Menu(root)

# Adding a cascade to the menu bar:
filemenu = Menu(menubar, tearoff=0)
filemenu_diagnosisMode_forFileDragDrop = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Files", menu=filemenu)

CONSTANT_DIAGNOSIS_IMAGE_SPAN = 480



def loadCovid19ImageFromDialog():
    currdir = os.getcwd()
    image_file = filedialog.askopenfile(mode='r', parent=root, initialdir=currdir,
                                        title='Please select an Xray Image of suspected coronavirus2019 case:')
    root.wm_title(windowTitle + " : " + image_file.name)
    loadCovid19ImageFromName(image_file.name)


def loadCovid19ImageFromName(filename):
    app._PRIOR_IMAGE.destroy()  # destroy old image
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),
                       Image.ANTIALIAS)  # Resized "load" image to constant size on screen. However, neural network still runs on on original image scale from filename.
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth) / 2) - CONSTANT_DIAGNOSIS_IMAGE_SPAN / 2,
              y=((int(screenHeight) / 2)) - CONSTANT_DIAGNOSIS_IMAGE_SPAN / 2 - 80)
    app.DIAGNOSIS_RESULT += "**Covid19 Mode Result**\n" + filename + "\n\n"
    app.DIAGNOSIS_RESULT += covid_diagnosis.doOnlineInference_covid19Pneumonia(filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img  # set latest instance of old image
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableDiagnosisResultColouring()


filemenu.add_command(label="Load Covid19 Pneumonia image", command=loadCovid19ImageFromDialog)

########################################################
# Drag drop functionality
# To specify which diagnosis mode drag and drop action should peform

checkbox_Covid19_ModeVar = IntVar()
Checkbutton(app, text="Activate Covid19 Mode (For Drag Drop)", variable=checkbox_Covid19_ModeVar).place(
    x=(int(screenWidth) / 2) - 136, y=24)




def setCovid19ImageToWindow_ViaDragAndDrop(filename):
    app._PRIOR_IMAGE.destroy()  # destroy old image
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),
                       Image.ANTIALIAS)  # Resized "load" image to constant size on screen. However, neural network still runs on on original image scale from filename.
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth) / 2) - CONSTANT_DIAGNOSIS_IMAGE_SPAN / 2,
              y=((int(screenHeight) / 2)) - CONSTANT_DIAGNOSIS_IMAGE_SPAN / 2 - 80)
    app.DIAGNOSIS_RESULT += "**Covid19 Mode (Drag & Drop) Result** \n" + filename + "\n\n"
    app.DIAGNOSIS_RESULT += covid_diagnosis.doOnlineInference_covid19Pneumonia(filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img  # set latest instance of old image
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableDiagnosisResultColouring()


def setSimultaneousRunImageToWindow_ViaDragAndDrop(filename):
    app._PRIOR_IMAGE.destroy()  # destroy old image
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),
                       Image.ANTIALIAS)  # Resized "load" image to constant size on screen. However, neural network still runs on on original image scale from filename.
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth) / 2) - CONSTANT_DIAGNOSIS_IMAGE_SPAN / 2,
              y=((int(screenHeight) / 2)) - CONSTANT_DIAGNOSIS_IMAGE_SPAN / 2 - 80)
    app.DIAGNOSIS_RESULT += "**Simultaneous Mode ~ Non-Covid19 & Covid19 (Drag & Drop) Result** \n" + filename + "\n"
    app.DIAGNOSIS_RESULT += "\n\n**Non-Covid19 Section** \n"
    #app.DIAGNOSIS_RESULT += covid19_ai_diagnoser.doOnlineInference_regularPneumonia(filename)
    app.DIAGNOSIS_RESULT += "**Covid19 Section** \n"
    app.DIAGNOSIS_RESULT += covid_diagnosis.doOnlineInference_covid19Pneumonia(filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img  # set latest instance of old image
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableSimultaneousDiagnosisResultColouring()


######################Jordan_update_drag_drop functionality
def drop(event):
    filename = event.data

    # Added OSError fix, using regex, to remove { or } symbols in filename from drag and drop library
    if ("{" in filename or "}" in filename):
        filename = re.search(r'\{(.*?)\}', filename).group(1)

    if (checkbox_Covid19_ModeVar.get() == 0):
        messagebox.showinfo(title="Diagnosis Drag & Drop Warning", message="No 'Drag & Drop' diagnosis mode selected!")
    elif (checkbox_Covid19_ModeVar.get() == 1):
        setCovid19ImageToWindow_ViaDragAndDrop(filename)



def colourDiagnosisMessageText(diagnosisContent, startIndexText, endIndexText):
    # If pneumonia or covid19 is detected
    if (covid_diagnosis.DIAGNOSIS_MESSAGES[0] in diagnosisContent or covid_diagnosis.DIAGNOSIS_MESSAGES[
        1] in diagnosisContent):
        app.DIAGNOSIS_RESULT_FIELD.tag_add("DIAGNOSIS_RESULT_MESSAGE_ABNORMAL", startIndexText, endIndexText)
        app.DIAGNOSIS_RESULT_FIELD.tag_configure("DIAGNOSIS_RESULT_MESSAGE_ABNORMAL", background="red",
                                                 foreground="white")

    # If normal lungs state is detected
    if (covid_diagnosis.DIAGNOSIS_MESSAGES[2] in diagnosisContent):
        app.DIAGNOSIS_RESULT_FIELD.tag_add("DIAGNOSIS_RESULT_MESSAGE_NORMAL", startIndexText, endIndexText)
        app.DIAGNOSIS_RESULT_FIELD.tag_configure("DIAGNOSIS_RESULT_MESSAGE_NORMAL", background="green",
                                                 foreground="white")


def enableDiagnosisResultColouring():
    diagnosisResultFieldContent = app.DIAGNOSIS_RESULT_FIELD.get("1.0", "end")
    colourDiagnosisMessageText(diagnosisResultFieldContent, "4.0", "4.21")


def enableSimultaneousDiagnosisResultColouring():
    diagnosisResultFieldContent_NonCovid19 = app.DIAGNOSIS_RESULT_FIELD.get("6.0", "7.0")
    diagnosisResultFieldContent_Covid19 = app.DIAGNOSIS_RESULT_FIELD.get("10.0", "11.0")
    colourDiagnosisMessageText(diagnosisResultFieldContent_NonCovid19, "6.0", "6.21")
    colourDiagnosisMessageText(diagnosisResultFieldContent_Covid19, "10.0", "10.21")


app.drop_target_register(DND_FILES)
app.dnd_bind('<<Drop>>', drop)

############
# root cycle
root.config(menu=menubar)
root.mainloop()
