import PySimpleGUI as sg
from bs4 import BeautifulSoup as bs
import pandas as pd
from ML import predict_CO2_emission
sg.theme("LightBlue7")


selection = ("Fuel Type_D","Fuel Type_E","Fuel Type_N","Fuel Type_X","Fuel Type_Z")

picture_column = sg.Column([
    [sg.Image('',key = '-IMAGE-', background_color = 'LightBlue', visible = False)]],
    key = '-LEFT-',

    background_color = '#FFFFFF')

info_column = sg.Column([
    [sg.Text('', key = '-CO2_Level-', font = 'Calibri 20', background_color = 'LightBlue', pad = 0, visible = False)],
    ],key = '-RIGHT-',
    background_color = '#FFFFFF')

main_layout = [
    [sg.Text('Fuel Type', size=(32, 1), font = 'Calibri 20',text_color= 'black'),],
    [sg.Combo(selection, size=(40, 10), font = 'Calibri 20', enable_events=True, key='-COMBO-')],
    [sg.Text('Engine Size(L)', size=(32, 1), font = 'Calibri 20',text_color= 'black')],
    [sg.Input(key = '-INPUT1-', font = 'Calibri 20',expand_x = True)],
    [sg.Text('Cylinders', font = 'Calibri 20', size=(32, 1),text_color= 'black')],
    [sg.Input(key = '-INPUT2-', font = 'Calibri 20',expand_x = True)],
    [sg.Text('Fuel Consumption Comb (L/100 km)', font = 'Calibri 20', size=(32, 1),text_color= 'black')],
    [sg.Input(key = '-INPUT3-', font = 'Calibri 20',expand_x = True)],
    [sg.Text('Fuel Consumption Comb (mpg)', font = 'Calibri 20', size=(32, 1),text_color= 'black')],
    [sg.Input(key = '-INPUT4-', font = 'Calibri 20',expand_x = True)],
    [sg.Button('submit', font = 'Calibri 20', button_color = 'black')],
    [info_column],
    [picture_column]
  
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
        print(show_data)
        if show_data == "[1]":
            window['-CO2_Level-'].update("CO2_Level 1 less than 199 ", visible = True)
            window['-IMAGE-'].update(filename = 'maimoke_level_1.png',visible = True)
            print('yeah')
        if show_data == "[2]":
            window['-CO2_Level-'].update("CO2_Level 2 range 199-227 ", visible = True)
            window['-IMAGE-'].update(filename = 'maimoke_level_2.png',visible = True)
            print('yeah')
        if show_data == "[3]":
            window['-CO2_Level-'].update("CO2_Level 3 range 227-254 ", visible = True)
            window['-IMAGE-'].update(filename = 'maimoke_level_3.PNG',visible = True)
            print('yeah')
        if show_data == "[4]":
            window['-CO2_Level-'].update("CO2_Level 4 range 254-290 ", visible = True)
            window['-IMAGE-'].update(filename = 'maimoke_level_4.png',visible = True)
            print('yeah')
        if show_data == "[5]":
            window['-CO2_Level-'].update("CO2_Level 5 more than 290 ", visible = True)
            window['-IMAGE-'].update(filename = 'maimoke_level_5.png',visible = True)
            print('yeah')


 
    
 
window.close()