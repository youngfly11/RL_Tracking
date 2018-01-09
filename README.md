# Env v1.3.2
### use hist_v1.3
* resize bg_box area 
* add gaussian filter and medium filter to improve response map and color map)

### change reward
* reward use precision( if dis(ground_truth, target) > 40px return 0 else return 1 - (dis / 40px))

### add file save function
* add file save function in class Env

### fix bugs
* fix bugs in show_tracking_result