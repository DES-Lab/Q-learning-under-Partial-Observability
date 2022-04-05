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
Or install each dependency individually in case if you want to use your GPU for comparison.
```
pip install aalpy
pip install stable-baselines
pip install tensorflow==1.15 (or tensorflow-gpu==1.15)
pip install numpy==1.16.4
pip install gym==0.15.7
```

To make Alergia faster, we interface to [jAlergia](https://github.com/emuskardin/jAlergia/tree/master) with AALpy.
Ensure that you have java added to the path.
If you have Java >= 12, provided `alergia.jar` should work out of the box.
If you have lower version of Java added to your path, please compile your own .jar file and replace the one present in the reposatory.
```
git clone https://github.com/emuskardin/jAlergia
gradlew jar
# gradlew.bat on Windows
```

## Approximate POMDP with finite-state deterministic MDP
To see an example how active or passive automata learning methods can be used to approximate a POMDP run:
```
python pomdp_approximation_demo.py
```

## Run experiments
To run each experiment, simply call call the appropriate python script with experiment name.
Experiment names found in paper are `oficeWorld`, `confusingOfficeWorld`, `gravity`, `thinMaze`.

```
python partially_observable_q_learning.py <exp_name>
python reccurent_policy_comp.py <exp_name>
python stacked_frames_comp.py <exp_name>
```