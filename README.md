# Deep Modeling for Molecular Simulation 2023
## Four-day in-person workshop, July 11-14, 2023

Welcome to the repository of the workshop! Here you will find the instructions for the tutorials and other useful material.

### Acknowledgements
This workshop was funded and organized by the center [Chemistry in Solution and at Interfaces](https://ccsc.princeton.edu/) funded by the U.S. Department of Energy under award DE-SC0019394. 


### Trouble Shooting
- Virtual machine cannot connect
Sometimes when you try to log into the virtual machine using `ssh`, the following error message may appear:
```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ED25519 key sent by the remote host is
SHA256:y0Lgjr7d7lw+rQnVXUU7JknCq2JzYrUDcMib0CBRUlA.
Please contact your system administrator.
Add correct host key in /Users/yifan/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /Users/yifan/.ssh/known_hosts:92
Host key for [lab-29e723ef-f8c9-4641-9982-3966bdf511db.eastus.cloudapp.azure.com]:5006 has changed and you have requested strict checking.
Host key verification failed.
```

If you see this error, you can delete the last 3 lines (or sometime 2 lines) that starts with `[lab-....eastus.cloudapp.azure.com]:` (execute these commands line by line):
```
vi ~/.ssh/known_hosts
Shift+g
2k
3dd
:wq
Enter
```