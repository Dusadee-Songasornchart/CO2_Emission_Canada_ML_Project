import PySimpleGUI as sg
from bs4 import BeautifulSoup as bs
import pandas as pd

df = pd.read_csv('X.csv')

selection = (df.columns[0],df.columns[1])


print(df)

main_layout = [
    [sg.Combo(selection, size=(20, 5), enable_events=True, key='-COMBO-')],
    [sg.Input(key = '-INPUT-',expand_x = True),sg.Button('submit', button_color = 'black')],
    
  
]
 
sg.theme('reddit')
window = sg.Window('Weather', main_layout)
combo = window['-COMBO-']
 
while True:
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED:
        break
    if event == 'summit':
        True
    if event == "-COMBO-":
        combo.Widget.event_generate('<Button-1>')

 
    
 
window.close()