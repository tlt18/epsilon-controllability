import numpy as np
import matplotlib.pyplot as plt

 
def plotDOC(datadirs, labels, env, max_step):
    # Set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Define line styles and markers
    line_styles = ['-', '--', '-.', ':']
    colors = ['red', 'cornflowerblue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
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
        plt.plot(step, doc, label=label, linestyle=line_styles[i % len(line_styles)], linewidth=2, color=colors[i % len(colors)])
        
    plt.xlabel("Steps",fontsize=14, fontname='Times New Roman')
    plt.ylabel("DOC",fontsize=14, fontname='Times New Roman')
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(f"./epsilon-controllability/figs/{env}/{env}_doc.pdf", format='pdf')
    
if __name__ == "__main__":
    config = {
        "MassSpring": {
            "dir_name": [
                "[0. 0.]state-0.05epsilon-5000samples-max_radius20240823-171803",
                "[0. 0.]state-0.05epsilon-5000samples-DFS20240823-171855",
                "[0. 0.]state-0.05epsilon-5000samples-BFS20240823-171819",
            ],
            "max_step": 15000,
        },
        "Oscillator": {
            "dir_name": [
                "[0. 0.]state-0.05epsilon-5000samples-max_radius20240823-172014",
                "[0. 0.]state-0.05epsilon-5000samples-DFS20240823-172426",
                "[0. 0.]state-0.05epsilon-5000samples-BFS20240823-172038",
            ],
            "max_step": 30000,
        },
        "TunnelDiode": {
            "dir_name": [
                "[0.8844298  0.21038036]state-0.1epsilon-5000samples-max_radius20240823-203709",
                "[0.8844298  0.21038036]state-0.1epsilon-5000samples-DFS20240823-213034",
                "[0.8844298  0.21038036]state-0.1epsilon-5000samples-BFS20240823-203946",
            ],
            "max_step": 30000,
            # "dir_name": [
            #     "[0.06263583 0.75824183]state-0.1epsilon-5000samples-max_radius20240823-202738",
            #     "[0.06263583 0.75824183]state-0.1epsilon-5000samples-DFS20240823-203240",
            #     "[0.06263583 0.75824183]state-0.1epsilon-5000samples-BFS20240823-202755",
            # ],
            # "max_step": 30000,
        }
    }
    
    env = "TunnelDiode"
    dir_name = config[env]["dir_name"]
    max_step = config[env]["max_step"]
    datadirs = [f"./epsilon-controllability/figs/{env}/{name}/count.txt" for name in dir_name]
    labels = ["MECS", "DFS", "BFS"]
    plotDOC(datadirs, labels, env, max_step)