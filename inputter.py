import keyboard
import re 
string = "up"


class ReplayAgent:
    
    
    def __init__(self,agent_name):
        self.name = agent_name
        self.high_score = ReplayAgent.get_high_score(self.name)



    @staticmethod
    def get_high_score(agent_name):
        regex = re.compile(f'{agent_name}')
        f = open("score.txt","r").readlines()
        for line in f:
            match = regex.match(line)
            if match:
                # print(line)
                return line

for _ in range(10):
    string +=",up"
keyboard.add_hotkey('down', lambda: keyboard.press(string))

keyboard.wait()


def main():
    TrentAgent = ReplayAgent("Trent")
    print(TrentAgent.high_score)
    
    
if __name__ == "__main__":
    main()
    


