import typer
from pathlib import Path
import PySimpleGUI as sg

def make_empty_dirs(base_dir: Path, starting_num=2, ending_num=2):
    """Make empty directories for each number in the range of starting_num to ending_num"""
    while starting_num <= ending_num:
        (base_dir / f"{starting_num}").mkdir(exist_ok=True)
        starting_num = starting_num ** 2
        print("hi")


def gui():
    sg.theme('DarkBlue')

    layout = [
        [sg.Text('Base Dir'), sg.InputText(key="BASE-DIR-DISPLAY"), sg.FolderBrowse(key="BASE-DIR-BROWSE")],
        [sg.Text('Start Number'), sg.InputText(key="START-NUM")],
        [sg.Text('End Number'), sg.InputText(key="END-NUM")],
        [sg.Button('Go'), sg.Button('Exit')]]

    window = sg.Window('LLPS Analysis', layout)

    while True:  # Event Loop
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go':
            make_empty_dirs(Path(values["BASE-DIR-DISPLAY"]), starting_num=int(values["START-NUM"]), ending_num=int(values["END-NUM"]))


if __name__ == "__main__":
    typer.run(gui)