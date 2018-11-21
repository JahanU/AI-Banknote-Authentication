To run the program:
Have both the dataset (which is in a CSV format called “Banknote_authentication”) 
and the python file (called “MachineLearningModel”) in the same folder or directory and
then run the python code with your preferred IDE.. 

NOTE:
The code should be ran in an IDE (such as Spyder) as there can issues with imported packages.
when trying to run the python file through command line, as shown below:

D:\Users\Jahan\Documents\Uni\Advanced AI COMP211\A1.1>py MachineLearningModel.py
Traceback (most recent call last):
  File "MachineLearningModel.py", line 28, in <module>
    import pandas as pd
  File "C:\Users\Jahan\Anaconda3\lib\site-packages\pandas\__init__.py", line 19, in <module>
    "Missing required dependencies {0}".format(missing_dependencies))
ImportError: Missing required dependencies ['numpy']

D:\Users\Jahan\Documents\Uni\Advanced AI COMP211\A1.1>

I do believe the following issue to be a case by case issue, but in the event it does not run through
command line, I would reccomend using Spyder IDE.

On windows, you would open the directory of the folder, and do the following:
py MachineLearningModel.py

On MAC:
python MachineLearningModel.py