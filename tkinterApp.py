import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import os
import io

#load model and define counters and start root app
model = tf.keras.models.load_model('mymodel/mymodel')
line_id = None
line_points = []
line_options = {}
root = tk.Tk()

#drawing line event that extend the line from previous point
def draw_line(event):
    global line_id
    line_points.extend((event.x, event.y))
    if line_id is not None:
        canvas.delete(line_id)
    line_id = canvas.create_line(line_points, **line_options,width=7)

#start the line
def set_start(event):
    line_points.extend((event.x, event.y))

#when the drawing is done create an image and pass through the model creating an messagebox about the answer
#then clear the drawing
def end_line(event=None):
    global line_id
    line_points.clear()
    line_id = None

    ps = canvas.postscript(colormode = 'mono', pageheight = 100, pagewidth=140)
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img.save('tmp.png')
    z = tf.keras.preprocessing.image.load_img('tmp.png', target_size=(100,140),color_mode='grayscale')
    os.remove('tmp.png')
    z = tf.keras.preprocessing.image.img_to_array(z)
    z = np.expand_dims(z, axis=0)
    z = np.array(z)
    y = model.predict(z)
    ysoft = tf.nn.softmax(y)
    ymax = np.argmax(ysoft)
    messagebox.showinfo("my guess is", ymax)
    canvas.delete('all')



#set the canvas background to pure white
canvas = tk.Canvas(bg='white')
canvas.pack()

#bind events to movement of mouse
canvas.bind('<Button-1>', set_start)
canvas.bind('<B1-Motion>', draw_line)
canvas.bind('<ButtonRelease-1>', end_line)

#start the app
root.mainloop()