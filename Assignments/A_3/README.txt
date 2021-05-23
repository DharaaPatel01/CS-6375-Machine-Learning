This assignment folder contains 4 files, 
Naive.py
NaiveWithoutStop.py
Logistic.py
LogisticWithoutStop.py

NOTE: - I was using MacOs so you might get error for paths. Please change ‘/spam’ to ‘\spam’ and likewise for paths for ham, train and test in Naive.py / NaiveWithoutStop.py files.

To run Naive.py/NaiveWithoutStop.py simply Run it in an IDE or terminal.

For Logistic.py/LogisticWithoutStop.py
In Spyder, use this command
runfile('<filename>.py', args='<filename(extra arg for when run in terminal, any dummy value will do)> <TRAIN_path> <TEST_path> <Lambda value> <No of Iterations>')

Ex:
runfile('/Users/raa/Documents/Spring\'21/CS 6375_ML/Assignments/A_3/Logistic.py', args='fileName-in-terminal /Users/raa/Documents/Spring\'21/CS\ 6375_ML/Assignments/A_3/train /Users/raa/Documents/Spring\'21/CS\ 6375_ML/Assignments/A_3/test 0.001 100')

In Terminal(CLE in Mac)

python <filename.py> <TRAIN_path> <TEST_path> <Lambda value> <No of Iterations>'

Ex:
python Logistic.py /Users/raa/Documents/Spring\'21/CS\ 6375_ML/Assignments/A_3/train /Users/raa/Documents/Spring\'21/CS\ 6375_ML/Assignments/A_3/test 1 100