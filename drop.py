from tkinter import *
from tkinter import ttk
import tkinter.font as font


root=Tk()
root.title("SANSKRIT TO MALAYALAM TRANSLATOR")
root.geometry("800x500")
root.configure(bg="#8c7e7b")
#root.eval('tk::PlaceWindow . center')


myFont = font.Font(family='Helvetica')


#drop down boxes

def show():
	return





options1=list(range(1,19))
options2=list(range(1,101))


l1=Label(root,text="chapter",font=myFont,padx=50)
l1.place(x=300,y=300)

#clicked=IntVar()
#clicked.set("chapter")

#drop=OptionMenu(root,clicked,*options)
#drop.pack(pady=20)

myCombo=ttk.Combobox(root,value=options1,width=10,height=20)
myCombo.place(x=500,y=300)

l2=Label(root,text="sloka",font=myFont,padx=50)
l2.place(x=800,y=300)

myCombo=ttk.Combobox(root,value=options2,width=10)
myCombo.place(x=1000,y=300)


myButton=Button(root,text="translate",font=myFont,command=show,padx=50)
myButton.place(x=600,y=400)

my_text=Text(root,width=40,height=10)
my_text.place(x=530,y=500)
root.mainloop()