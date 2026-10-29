"""Opt-in capture of NumPy cast-overflow context for a single subprocess.

Only activated when this directory is explicitly added to PYTHONPATH and an
output path is supplied. Does not change Warp, NumPy, or the warning behavior.
"""
import json
import os
import traceback
import warnings

_output=os.environ.get('SUPERMIMIC_WARNING_AUDIT')
_original=warnings.showwarning


def _showwarning(message,category,filename,lineno,file=None,line=None):
    if _output and issubclass(category,RuntimeWarning) and 'overflow encountered in cast' in str(message):
        frames=[];frame=__import__('sys')._getframe()
        warp_local=None
        while frame is not None:
            code=frame.f_code
            frames.append({'file':code.co_filename,'line':frame.f_lineno,'function':code.co_name})
            if warp_local is None and code.co_filename.replace('\\','/').endswith('/warp/_src/types.py') and 'data' in frame.f_locals:
                warp_local=frame.f_locals
            frame=frame.f_back
        record={'category':category.__name__,'message':str(message),'warning_file':filename,
                'warning_line':lineno,'stack':frames[:25]}
        if warp_local is not None:
            data=warp_local['data'];record['source_type']=type(data).__name__
            record['source_shape']=list(data.shape) if hasattr(data,'shape') else None
            record['source_dtype']=str(data.dtype) if hasattr(data,'dtype') else None
            record['target_dtype']=str(warp_local.get('npdtype'))
            if type(data).__module__.startswith('numpy') and getattr(getattr(data,'dtype',None),'kind','')=='f':
                import numpy as np
                finite=np.isfinite(data)
                record['finite_count']=int(finite.sum());record['nonfinite_count']=int(data.size-finite.sum())
                record['over_float32_count']=int((finite & (np.abs(data)>np.finfo(np.float32).max)).sum())
                record['max_abs_finite']=float(np.max(np.abs(data[finite]))) if finite.any() else None
        try:
            with open(_output,'a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        except Exception as exc:
            __import__('sys').stderr.write(f'warning audit failed: {type(exc).__name__}\n')
    return _original(message,category,filename,lineno,file,line)


if _output:
    warnings.simplefilter('always',RuntimeWarning)
    warnings.showwarning=_showwarning
