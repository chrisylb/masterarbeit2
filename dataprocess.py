import pandas as pd
import numpy as np
import math
actions=np.zeros((33000,1,2))
rewards=np.zeros((33000,1,1))
observations=np.zeros((33000,1,4,4))
pos=0
pos1=0
pos2=0
next_observations=np.zeros((33000,1,4,4))
next_observations_id=[0]
ogdata=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dataprocess/vehicle_tracks_011.csv')
translate_angle_deger=3.064*180/np.pi
yaws=(180-translate_angle_deger)*np.pi/180
R=np.array([[np.cos(yaws),-np.sin(yaws)],[np.sin(yaws),np.cos(yaws)]])
ogdata.loc[:,'x_trans']=R[0][0]*ogdata.loc[:,'x']+R[0][1]*ogdata.loc[:,'y']
ogdata.loc[:,'y_trans']=R[1][0]*ogdata.loc[:,'x']+R[1][1]*ogdata.loc[:,'y']
ogdata.loc[:,'vx_trans']=R[0][0]*ogdata.loc[:,'vx']+R[0][1]*ogdata.loc[:,'vy']
ogdata.loc[:,'vy_trans']=R[1][0]*ogdata.loc[:,'vx']+R[1][1]*ogdata.loc[:,'vy']
studydata=ogdata.drop(['x','y','vx','vy'],axis=1)
studydata['x_trans']=studydata['x_trans'].map(lambda x:1000-x)
studydata['vx_trans']=studydata['vx_trans'].map(lambda x:-x)
studydata['y_trans']=studydata['y_trans'].map(lambda y:1100-y)
studydata['vy_trans']=studydata['vy_trans'].map(lambda y:-y)
def merge_y(a):
    if a.y_trans>20:
        a.y_trans=None
    return a
merge=studydata.apply(merge_y,axis='columns')
Mergeid=merge.dropna(subset=['y_trans'])
id=Mergeid['track_id'].unique()

for xyz in range(0, len(id)):

    def find_frame(d):
        if not d.track_id == id[xyz]:
            d.track_id = None
        return d


    studycar = studydata.apply(find_frame, axis='columns')
    studycar = studycar.dropna(subset=['track_id'])
    studycar = studycar[studycar.x_trans > 80]
    studycar = studycar[studycar.x_trans < 156]
    studycar.loc[:, 'x_acclearation'] = 10 * (studycar.loc[:, 'vx_trans'].shift(-1) - studycar.loc[:, 'vx_trans'])
    studycar.loc[:, 'y_acclearation'] = 10 * (studycar.loc[:, 'vy_trans'].shift(-1) - studycar.loc[:, 'vy_trans'])
    studycar.loc[:, 'speed'] = np.sqrt(
        (studycar.loc[:, 'vx_trans'] ** 2) + (studycar.loc[:, 'vy_trans'] ** 2))
    studycar.loc[:, 'accleration'] = 10 * (studycar.loc[:, 'speed'].shift(-1) - studycar.loc[:, 'speed'])
    studycar = studycar.fillna(method='ffill')
    studycar.loc[:, 'heading'] = 0.00
    studycar.loc[:, 'steering_angle'] = 0.00
    studycar.loc[:, 'beta'] = 0.00
    studycar = studycar.reset_index(drop=True)
    y_1 = studycar.iloc[1].y_trans - studycar.iloc[0].y_trans
    x_1 = studycar.iloc[1].x_trans - studycar.iloc[0].x_trans

    studycar.loc[0, 'heading'] = np.arctan(y_1 / x_1)

    for h in range(1, len(studycar)):
        if studycar.loc[h - 1, 'vy_trans'] == 0:
            studycar.loc[h, 'total_angle'] = studycar.loc[h - 1, 'heading']
        else:
            studycar.loc[h, 'total_angle'] = np.arctan(
                studycar.loc[h - 1, 'vy_trans'] / studycar.loc[h - 1, 'vx_trans'])
        beta = studycar.loc[h, 'total_angle'] - studycar.loc[h - 1, 'heading']
        studycar.loc[h, 'steering_angle'] = np.arctan(2 * np.tan(beta))
        studycar.loc[h, 'heading'] = studycar.loc[h - 1, 'speed'] * np.sin(beta) / (4.28 / 2) * 0.1 + studycar.loc[
            h - 1, 'heading']

    # studycar = studycar.fillna(method='ffill')
    studycar.loc[:, 'reward'] = 0
    goal = False
    for g in range(0, len(studycar) - 1):
        if studycar.loc[g + 1, 'x_trans'] > 155 and goal == False:
            studycar.loc[g + 1, 'reward'] = 10 + studycar.loc[g, 'reward']
            goal = True
        else:
            studycar.loc[g + 1, 'reward'] = 0.01 + studycar.loc[g, 'reward']
    # studycar is the merge car and frame_car is the car which is driving with the merge_ego_car

    frame_id = studycar.frame_id.unique()
    frame_car = studydata.set_index(['frame_id'])
    frame_car = frame_car.loc[frame_id]


    def find_otherframe(c):
        if c.track_id == id[xyz]:
            c.track_id = None
        return c


    frame_car = frame_car.apply(find_otherframe, axis='columns')
    frame_car = frame_car.dropna(subset=['track_id'])

    #####

    study = studycar.set_index(['frame_id'])
    for index, row in studycar.iterrows():
        observations[pos][0][0][0] = studycar.loc[index, 'x_trans']
        observations[pos][0][0][1] = studycar.loc[index, 'y_trans']
        observations[pos][0][0][2] = studycar.loc[index, 'vx_trans']
        observations[pos][0][0][3] = studycar.loc[index, 'vy_trans']
        pos = pos + 1
    # 10 safe distance
    for k in range(0, len(studycar)):
        bd = 0
        if frame_id[k] in frame_car.index:
            if not isinstance(frame_car.loc[frame_id[k], 'x_trans'], np.float64):

                for v in range(0, min(len(frame_car.loc[frame_id[k]]), 3)):
                    if np.abs(frame_car.loc[frame_id[k], 'x_trans'].iloc[v] - study.loc[frame_id[k], 'x_trans']) < 10:
                        observations[k + pos1][0][bd + 1][0] = frame_car.loc[frame_id[k], 'x_trans'].iloc[v]
                        observations[k + pos1][0][bd + 1][1] = frame_car.loc[frame_id[k], 'y_trans'].iloc[v]
                        observations[k + pos1][0][bd + 1][2] = frame_car.loc[frame_id[k], 'vx_trans'].iloc[v]
                        observations[k + pos1][0][bd + 1][3] = frame_car.loc[frame_id[k], 'vy_trans'].iloc[v]
                        bd = bd + 1
            else:
                observations[k + pos1][0][bd + 1][0] = frame_car.loc[frame_id[k], 'x_trans']
                observations[k + pos1][0][bd + 1][1] = frame_car.loc[frame_id[k], 'y_trans']
                observations[k + pos1][0][bd + 1][2] = frame_car.loc[frame_id[k], 'vx_trans']
                observations[k + pos1][0][bd + 1][3] = frame_car.loc[frame_id[k], 'vy_trans']
    pos1 = pos1 + len(studycar)

    for p in range(0, len(studycar)):
        actions[p + pos2] = [studycar.loc[p, 'accleration'], studycar.loc[p, 'steering_angle']]

    for j in range(0, len(studycar)):
        rewards[j + pos2] = [studycar.loc[j, 'reward']]

    pos2 = pos2 + len(studycar)
    next_observations_id.append(pos2 - 1)
    studycar_describe.append(studycar.y_trans.min())
    print(pos1)

np.save('actions_interaction_4',actions)
np.save('obs_interaction_4',observations)
np.save('rewards_interaction_4',rewards)
for i in range(0,29999):
  if i not in next_observations_id:
    next_observations[i]=observations[i+1]
  else:
    next_observations[i]=observations[i]
np.save('next_obs_interaction_4',observations)
