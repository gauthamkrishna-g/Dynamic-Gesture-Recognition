uWave Gesture Library
Author:	Rice Efficient Computing Group, Rice University
Date:		March, 2009


Copyright and license

1. We grant you a nonexclusive, nontransferable license to use the data for commercial, educational, and/or research purposes only. You agree to not redistribute the data without written permission from us. 

2. You agree to acknowledge the source of the data, i.e., the uWave project, by citing the following paper in your publication or product.

Jiayang Liu, Zhen Wang, Lin Zhong, Jehan Wickramasuriya, and Venu Vasudevan, "uWave: Accelerometer-based personalized gesture recognition and its applications," in Proc. IEEE Int. Conf. Pervasive Computing and Communication (PerCom), March 2009.

3. We provide no warranty whatsoever on any aspect of the data. Use at your own risk.


The following is a description of the file ogranization of uWave gesture library.

On the top level, each .rar file includes the gesture samples collected from one user on one day. 
The .rar files are named as U$userIndex ($dayIndex).rar, where $userIndex is the index of the participant from 1 to 8, and $dayIndex is the index of the day from 1 to 7.

Inside each .rar file, there are .txt files recording the time series of acceleration of each gesture. 
The .txt files are named as *$gestureIndex-$repeatIndex.txt, where $gestureIndex is the index of the gesture as in the 8-gesture vocabulary, and $repeatIndex is the index of the repitition of the same gesture pattern from 1 to 10.

In each .txt file, the first column is the x-axis acceleration, the second y-axis acceleration, and the third z-axis acceleration. 
The unit of the acceleration data is G, or acceleration of gravity.



