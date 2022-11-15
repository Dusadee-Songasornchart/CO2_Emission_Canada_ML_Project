import PySimpleGUI as sg
from bs4 import BeautifulSoup as bs
import pandas as pd
from ML import predict_CO2_emission



selection = ("Fuel Type_D","Fuel Type_E","Fuel Type_N","Fuel Type_X","Fuel Type_Z")



info_column = sg.Column([
    [sg.Text('', key = '-CO2_Level-', font = 'Calibri 30', background_color = '#FF0000', pad = 0, visible = False)],
    ],key = '-RIGHT-',
    background_color = '#FFFFFF')

main_layout = [
    [sg.Text('Fuel Type', size=(26, 1),text_color= 'white')],
    [sg.Combo(selection, size=(40, 10), enable_events=True, key='-COMBO-')],
    [sg.Text('Engine Size(L)', size=(26, 1),text_color= 'white')],
    [sg.Input(key = '-INPUT1-',expand_x = True)],
    [sg.Text('Cylinders', size=(26, 1),text_color= 'white')],
    [sg.Input(key = '-INPUT2-',expand_x = True)],
    [sg.Text('Fuel Consumption Comb (L/100 km)', size=(26, 1),text_color= 'white')],
    [sg.Input(key = '-INPUT3-',expand_x = True)],
    [sg.Text('Fuel Consumption Comb (mpg)', size=(26, 1),text_color= 'white')],
    [sg.Input(key = '-INPUT4-',expand_x = True)],
    [sg.Button('submit', button_color = 'black')],
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
        data_0 = str(values['-COMBO-'])
        data_1 = float(values['-INPUT1-'])
        data_2 = int(values['-INPUT2-'])
        data_3 = float(values['-INPUT3-'])
        data_4 = int(values['-INPUT4-'])
        data_x = predict_CO2_emission(data_1,data_2,data_3,data_4,data_0)
        show_data = str(data_x.predict())
        window['-CO2_Level-'].update(show_data, visible = True)


 
    
 
window.close()