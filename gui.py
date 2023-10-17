import PySimpleGUI as sg

from main import main

sg.theme('DarkBlue')

layout = [
    [sg.Text('Input Folder'), sg.InputText(key="INPUT-FOLDER-DISPLAY"), sg.FolderBrowse(key="INPUT-FOLDER-BROWSE")],
    [sg.Text('Output Folder'), sg.InputText(key="OUPUT-FOLDER-DISPLAY"), sg.FolderBrowse(key="OUTPUT-FOLDER-BROWSE")],
    [sg.Text('Protein Name'), sg.InputText(key="METADATA")],
    [sg.Text('Method'), sg.Combo(key="METHOD", values=["STD", "OTSU"])],
    [sg.Button('Process'), sg.Button('Exit')]]

window = sg.Window('LLPS Analysis', layout)

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Process':
        window.perform_long_operation(lambda: main(values["INPUT-FOLDER-DISPLAY"], values["OUPUT-FOLDER-DISPLAY"],
                                                   metadata=[values["METADATA"], ],
                                                   method=values["METHOD"]), "FUNCTION_COMPLETED")

    if event == "FUNCTION_COMPLETED":
        sg.popup("Done!")

window.close()
