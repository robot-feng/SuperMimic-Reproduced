"""Streaming trace/covariance for paired full-vector gradient samples."""
import torch


class VectorMoments:
    def __init__(self):
        self.n=0
        self.mean_r=None
        self.mean_z=None
        self.m2_r=0.
        self.m2_z=0.
        self.cross=0.

    def add(self,r,z):
        if r.ndim!=1 or r.shape!=z.shape or not torch.isfinite(r).all() or not torch.isfinite(z).all():
            raise ValueError('invalid gradient vectors')
        r,z=r.detach(),z.detach()
        if self.n==0:
            self.mean_r=torch.zeros_like(r);self.mean_z=torch.zeros_like(z)
        elif self.mean_r.shape!=r.shape:
            raise ValueError('gradient size changed')
        self.n+=1
        dr=r-self.mean_r;dz=z-self.mean_z
        factor=(self.n-1)/self.n
        self.m2_r+=factor*float(torch.dot(dr,dr))
        self.m2_z+=factor*float(torch.dot(dz,dz))
        self.cross+=factor*float(torch.dot(dr,dz))
        self.mean_r.add_(dr,alpha=1/self.n);self.mean_z.add_(dz,alpha=1/self.n)

    def summary(self):
        if self.n<2:raise ValueError('need multiple vector samples')
        scale=self.n-1
        vr=self.m2_r/scale;vz=self.m2_z/scale;cov=self.cross/scale
        if vr<=0 or vz<=0:raise ValueError('degenerate vector variance')
        coef=-cov/vz
        return {'count':self.n,'real_variance_trace':vr,'zero_variance_trace':vz,'cross_covariance_trace':cov,
                'unit_cv_variance_trace':vr+2*cov+vz,'unit_cv_ratio':(vr+2*cov+vz)/vr,
                'fitted_coefficient':coef,'real_mean_norm':float(self.mean_r.norm()),'zero_mean_norm':float(self.mean_z.norm())}
