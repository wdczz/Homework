from rich.console import Console
from train import *
from predict import *
if __name__ == '__main__':
    console = Console()
    console.rule("Start")
    while True:
        down='0'
        choose = console.input("Choose your action --- 1. train the net / 2. use the net to predict / 3. Quit -> ")
        if choose=="1":
            train(console)
        elif choose == "2":
            pre(console)
        elif choose == "3":
            break
        else:
            console.log("You input a error number {}, so you can try to input 1 or 2!".format(choose))
            continue
        while not (down =="1" or down =='2') :
            down = console.input("Do you want to shut down this windows? if you want ,please enter 1.Otherwise enter 2 -> ")
            if down=="1":
                break
            elif down == "2":
                break
            else:
                console.log("You input a error number {}, so you can try to input 1 or 2!".format(choose))
        if down == "1":
            break
        elif down == "2":
            console.log("Go on!")

    console.rule("End")





