# Next Best View Algorithm : HERCULES I
## Version 1
Implementation of a next-best-view camera positioning algorithm, developed in the PyBullet environment and utilizing the PyVista library.


 <div style="text-align:center;">
    <img src="image/2.png" alt="2" width="800">
</div>

We create a folder and set up a Python virtual environment using the command:

```
python3 -m venv venv
```

You activate the virtual environment using the command:
```
source venv/bin/activate
```
We install the necessary libraries.

 And we open three parallel terminals. In each one, we run:

* In the 1st terminal, we type:

```
python3 world2.py
```

* In the 2nd terminal, we type:

```
python3 view_scanner_only_obj.py
```
* In the 3rd terminal, we type:

```
python3 edit_global_control.py
```

And once the three windows open, we press the `space key` to start the algorithm.

 <div style="text-align:center;">
    <img src="image/1.png" alt="1" width="800">
</div>

<div style="text-align:center;">
    <img src="image/3.png" alt="3" width="800">
</div>

## Version 2

 And we open three parallel terminals. In each one, we run:

* In the 1st terminal, we type:

```
python3 world2.py
```

* In the 2nd terminal, we type:

```
python3 view_scanner_only_obj_new_.py
```
* In the 3rd terminal, we type:

```
python3 edit_new4.py
```

And once the three windows open, we press the `space key` to start the algorithm.

 <div style="text-align:center;">
    <img src="image/4.png" alt="1" width="800">
</div>

<div style="text-align:center;">
    <img src="image/5.png" alt="3" width="800">
</div>


This repository contains a real-time robotic simulation environment built with PyBullet, featuring a UR5 robotic arm, a rotating turntable, and an end-effector–mounted virtual camera system for synthetic vision and 3D scene observation.

```
python3 world_with_ur5_turntable.py 
```

<div style="text-align:center;">
    <img src="image/6.png" alt="3" width="800">
</div>

