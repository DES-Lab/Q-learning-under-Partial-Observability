## Installation Process for Comparison
Due to the older version of tensorflow reacquired by stable baselines v.2, 
ensure that you have Python 3.6 installed.

With python 3.6 create the [virtual enviroment](https://python.land/virtual-environments/virtualent):
```
python3.6 -m venv .
source myvenv/bin/activate // Linux and Mac
myenv\Scripts\activate.bat // Windows
```

Install reacquired dependencies for comparison.
```
pip install stable-baselines
pip install tensorflow=1.15 (or tensorflow-gpu==1.15)
pip install numpy==1.16.4
pip install gym==0.15.7
```

Run comparison by defining the experiment in the bottom of the appropriate file and calling
```
python reccurent_policy_comp.py
python stacked_frames_comp.py
```
``