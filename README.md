## Installation Process for Comparison
Due to the older version of tensorflow recquired by stable baselines v.2, 
ensure that you have Python 3.6 installed. Hint: Create a virtual environment.
```
python 3.6
```
Stable baselines implements algorithms with recurrent policies against which we compare our method.
```
pip install stable-baselines
```
Tensorflow
```
pip install tensorflow=1.15
or
pip install tensorflow-gpu==1.15 
```
Gym version needs to be brought down to 0.15.7
```
pip install gym==0.15.7
```