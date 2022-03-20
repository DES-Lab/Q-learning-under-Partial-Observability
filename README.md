## Installation Process
Due to the older version of tensorflow reacquired by stable baselines v.2, ensure that you have Python 3.6 installed.
Note that our algorithm works with Python 3.6 and newer versions, but for the sake of comparison use Python3.6.

With python 3.6 create the [virtual enviroment](https://python.land/virtual-environments/virtualent):
```
python3.6 -m venv .
source myvenv/bin/activate // Linux and Mac
myenv\Scripts\activate.bat // Windows

python -m pip install --upgrade pip setuptools // to ensure that tensorflow 1.15 will be found
```

Install reacquired dependencies for comparison.
With a one liner
```
pip install -r recquirements.txt
```
Or install each dependecy individually in case if you want to use your GPU for comparison.
```
pip install aalpy
pip install stable-baselines
pip install tensorflow==1.15 (or tensorflow-gpu==1.15)
pip install numpy==1.16.4
pip install gym==0.15.7
```

## Comparison With LSTM-based policies and MlpPolicy with Frame stacking
Run comparison by defining the experiment in the bottom of the appropriate file and calling
```
python reccurent_policy_comp.py
python stacked_frames_comp.py
```
``