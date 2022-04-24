  GradientDescent-Algorithms
Gradient Descent Algorithms


## Dataset
515345 instances with 90 features of each instance <br>
<a> https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD </a>


## Model
Linear model where w is column vector with 90 entries and each X is 90 by 51345 matrix where each columns represent sample. <br>
our loss function is half MSE loss with L_2 regularization with parameter lambda / 2


## Train + Test
train : first 463,715 examples <br>
test : last 51,630 examples <br>


## Algorithms

1. Gradient Descent (GD).
2. Accelerated Gradient Descent (AGD).
3. Coordinate Descent - cyclic order approach (CDCO).
4. Coordinate Descent - random sampling approach (CDRS).

for more info go [here](https://github.com/tamirgabgab/GradientDescent-Algorithms/blob/main/IE598_BigDataOpt_lecturenotes_fall2016_lecture9_accelerated_GD.pdf) (module II)

## Results

part 1 results (choose setpsize = 1/L and setpsize = 2/(mu+L))
![m_merged](https://user-images.githubusercontent.com/80973047/164988913-13e8029b-8c8d-4424-9aa8-53656e339de2.png)

for setpsize = 1/L the best performance is AGD’s (as expected). <br>
for setpsize = 2/(mu+L) the best performance is GD’s (even better than the AGD!). <br>


part 2 results (AGD & CDCO comparision to uppre bound)

![m_merged (3)](https://user-images.githubusercontent.com/80973047/164988930-261643d4-63b9-46fc-a10f-a653840c654a.png)

It seems that the algorithm exceeds the bound in AGD 𝛽2, but it’s actually caused by the digital representation of the computer, one may call it ’approximation error’, which isn’t taken in count on the theoretical developments.


part 3 results (GD & CDRS comparision to uppre bound)

![m_merged (4)](https://user-images.githubusercontent.com/80973047/164988936-78a4817e-802f-4c40-ab7d-de126aac83d2.png)


## Files
### GradientDescent.py
### YearPredictionMSD.txt

## Dependencies
Python 3.8 <br>
pandas <br>
numpy <br>
matplotlib <br>
time <br>
