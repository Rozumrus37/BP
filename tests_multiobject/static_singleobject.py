#!/usr/bin/python

import vott as vot
import sys
import time

# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************
print(111)
handle = vot.VOT("rectangle")
print(112)
selection = handle.region()
print(113)

# Process the first frame
imagefile = handle.frame()
print(114)

if not imagefile:
    sys.exit(0)

print(115)

while True:
    print(6)
    # *****************************************
    # VOT: Call frame method to get path of the 
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************
    imagefile = handle.frame()
    if not imagefile:
        break

    # *****************************************
    # VOT: Report the position of the object 
    #      every frame using report method.
    # *****************************************
    handle.report(selection)

    time.sleep(0.01)


