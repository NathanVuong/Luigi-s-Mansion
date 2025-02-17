
import os
import matplotlib.pyplot as plt
 
class Parser(): 
    def __init__(self, job):
        self.job = job 
        self.time = []
        self.reward = []
        self.info = {}
        self.file = f"slurm-{job}.out"

    def parse_rewards(self):
        with open(self.file, "r") as f:
            for line in f:
                if line.startswith("Step output: "):
                    self.parse_step(line.strip().removeprefix("Step output: "))

    def parse_step(self, line):
        '''Helper for parse_rewards'''
        state, reward, done, trunc, info = line.split(',', 4)
        info = eval(info) #turns into dictionary
        self.time.append(info['time'])
        self.reward.append(float(reward))

    def graph(self):
        '''Visualizes information extracted'''
        plt.plot(self.time, self.reward, marker='o', color='b', label='Reward over Time')
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title('Time vs Reward')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def __repr__(self):
        '''Text representation'''
        return f'fix later'

    def process(self):
        '''Combines all steps'''
        self.parse_rewards()
        self.graph()


rewards = Parser(35858611)

def main():
    runs = {}
    #dir = input() #or manually input it
    dir = os.getcwd() #please call from logs directory
    for file in os.listdir(dir): 
        if file.startswith("slurm-") and file.endswith(".out"):
            job = int(file.removeprefix("slurm-").removesuffix(".out"))
            parse = Parser(job)
            parse.process()
    # for run in runs:
    #     run.parse_rewards()
    #     # print(run)

if __name__ == "__main__":    
    main()