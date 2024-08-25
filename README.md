# README

## Code Description

To run the code, first install Python3, and then install the required packages using the following command:
```sh
pip install -r requirement.txt
```

## Simulation Software Description

Run the following command to allow data collection, Lipschitz constant estimation, and controllability testing:
```sh
python main_test.py
```

## Experiment Design Description

To reproduce the experiments in the manuscript, you can set the parameters via Python's `argparse` and run the following commands:

```sh
# For CarFollowing environment
python main_test.py --env CarFollowing --num_sample 5000 --epsilon 0.05 --target_state 0.0 0.0

# For Oscillator environment
python main_test.py --env Oscillator --num_sample 5000 --epsilon 0.05 --target_state 0.0 0.0

# For TunnelDiode environment with different target states
python main_test.py --env TunnelDiode --num_sample 5000 --epsilon 0.05 --target_state 0.06263583 0.75824183
python main_test.py --env TunnelDiode --num_sample 5000 --epsilon 0.05 --target_state 0.8844298 0.210380361
```
