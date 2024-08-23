import numpy as np
import matplotlib.pyplot as plt

 
def plotDOC(datadirs, labels, env, max_step):
    # Set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Define line styles and markers
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    
    for i, (datadir, label) in enumerate(zip(datadirs, labels)):
        step = []
        doc = []
        with open(datadir, "r") as f:
            for line in f:
                line = line.strip().split(",")
                step.append(int(line[0]))
                doc.append(float(line[1]))
        index = np.where(np.array(step) <= max_step)
        step = np.array(step)[index]
        doc = np.array(doc)[index]
        
        # Use different line styles and markers
        plt.plot(step, doc, label=label, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)])
        
        plt.xlabel("Steps")
        plt.ylabel("DOC")
        
    plt.legend()
    plt.savefig(f"./epsilon-controllability/figs/{env}/{env}_doc.pdf", format='pdf')
    
if __name__ == "__main__":
    env = "MassSpring"
    dir_name = [
        "[0. 0.]state-0.05epsilon-5000samples-max_radius20240823-171803",
        "[0. 0.]state-0.05epsilon-5000samples-BFS20240823-171819",
        "[0. 0.]state-0.05epsilon-5000samples-DFS20240823-171855",
    ]
    max_step = 15000
    
    datadirs = [f"./epsilon-controllability/figs/{env}/{name}/count.txt" for name in dir_name]
    labels = ["MECS", "BFS", "DFS"]
    env = "MassSpring"
    plotDOC(datadirs, labels, env, max_step)