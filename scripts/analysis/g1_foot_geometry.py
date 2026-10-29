"""Geometric G1 foot-capsule clearance above a horizontal z=0 plane."""
from __future__ import annotations
import mujoco
import torch
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply,quat_mul

class FootClearance:
 def __init__(self,model,device='cpu',dtype=torch.float32):
  self.fk=G1Kinematics(model,device=device,dtype=dtype)
  ids=[];sides=[]
  for i in range(model.ngeom):
   name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
   side=0 if name.startswith('left_foot') else 1 if name.startswith('right_foot') else None
   if side is not None and name.endswith('_collision'):
    if model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_CAPSULE:raise ValueError(f'unsupported foot geometry {name}')
    ids.append(i);sides.append(side)
  if len(ids)!=14 or sides.count(0)!=7 or sides.count(1)!=7:raise ValueError('unexpected G1 foot capsules')
  self.ids=ids;self.sides=sides
  self.bodies=torch.tensor([int(model.geom_bodyid[i])-1 for i in ids],dtype=torch.long,device=device)
  self.pos=torch.tensor(model.geom_pos[ids],device=device,dtype=dtype)
  self.quat=torch.tensor(model.geom_quat[ids],device=device,dtype=dtype)
  self.radius=torch.tensor(model.geom_size[ids,0],device=device,dtype=dtype)
  self.half_length=torch.tensor(model.geom_size[ids,1],device=device,dtype=dtype)

 def __call__(self,state):
  body=self.fk(state[..., :36]);pos=body['pos'][...,self.bodies,:];quat=body['quat'][...,self.bodies,:]
  shape=pos.shape;local_pos=self.pos.expand(shape);local_quat=self.quat.expand(*shape[:-1],4)
  center=pos+quat_apply(quat,local_pos)
  geom_quat=quat_mul(quat,local_quat);axis=torch.zeros_like(center);axis[...,2]=1
  axis=quat_apply(geom_quat,axis)
  bottom=center[...,2]-self.half_length*axis[...,2].abs()-self.radius
  left=bottom[...,torch.tensor(self.sides,device=state.device)==0].amin(-1)
  right=bottom[...,torch.tensor(self.sides,device=state.device)==1].amin(-1)
  return torch.stack((left,right),-1)
