import PySimpleGUI as sg
from bs4 import BeautifulSoup as bs
import pandas as pd

df_X = pd.read_csv('X.csv')
df_Y = pd.read_csv('Y.csv')

df_X.drop(columns=['Unnamed: 0'],inplace= True)
df_Y.drop(columns=['Unnamed: 0'],inplace= True)
print(df_X)
print(df_Y)

selection = (df_X.columns[4],df_X.columns[5],df_X.columns[6],df_X.columns[7],df_X.columns[8])



info_column = sg.Column([
    [sg.Text('', key = '-TEST-', font = 'Calibri 30', background_color = '#FF0000', pad = 0, visible = False)]
    ],key = '-RIGHT-',
    background_color = '#FFFFFF')

main_layout = [
    [sg.Combo(selection, size=(20, 5), enable_events=True, key='-COMBO-'),sg.Button('submit', button_color = 'black')],
    [info_column]
  
]
 


sg.theme('reddit')
window = sg.Window('CO2_Emission', main_layout)

 
while True:
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED:
        break
    if event == 'submit':
        data = str(values['-COMBO-'])
        print(data)
        window['-TEST-'].update(data, visible = True)


 
    
 
window.close()