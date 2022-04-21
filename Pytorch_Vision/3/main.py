from rich.console import Console
from train import *
from predict import *
if __name__ == '__main__':
    console = Console()
    console.rule("Start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print("Your device is {}".format(device))
    while True:
        down='0'
        choose = console.input("Choose your action --- 1. train the net / 2. use the net to predict / 3. Quit -> ")
        if choose=="1":
            net_choose = console.input("Choose your net to train  --- 1. Convnet / 2. Lenet / -> ")
            epoch = console.input("Choose how many epoch to train your net  -> ")
            train(console,net_choose,device,epoch)
        elif choose == "2":
            while True:
                net_choose = console.input("Choose your net to predict  --- 1. Convnet / 2. Lenet / -> ")
                if net_choose=="1" or net_choose=="2":
                    pre(console,net_choose,device)
                    break
                else:
                    console.print("the number is not I need ,so try again!")
                    continue
        elif choose == "3":
            break
        else:
            console.log("You input a error number {}, so you can try to input 1 or 2!".format(choose))
            continue
        while not (down =="quit" or down =='again') :
            down = console.input("Do you want to shut down this windows? if you want to quit ,please enter <quit> .Go on enter <again> -> ")
            if down=="quit":
                break
            elif down == "again":
                break
            else:
                console.log("You input a error number {}, so you can try to input <quit> or <again>!".format(choose))
        if down == "quit":
            break
        elif down == "again":
            console.log("Go on!")

    console.rule("End")





