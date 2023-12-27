import keyboard

def main():
    while True:
        if keyboard.is_pressed('a'):
            print("A pressed - move left")
        elif keyboard.is_pressed('d'):
            print("D pressed - move right")
        elif keyboard.is_pressed('w'):
            print("W pressed - jump")
        else:
            print("No valid key pressed")

if __name__ == "__main__":
    main()
