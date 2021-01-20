import numpy as np
import h5py

def pclip(ref_p,pclip=98):
	a = -np.quantile(np.abs(ref_p[:]),(pclip/100))
	b = -a
	vmin=a
	vmax=b
	return vmin, vmax

def load_all_models(h5file_path):
    """Extract 2D models from h5 file into 3D numpy arrays [index, x, z].    
    Input:
    - h5file_path [str]
    Return: 
    - Ref_models, True_models, Smooth_models  [np.array]    
    """
    print('Loading data')
    
    with h5py.File(h5file_path,'r') as trModels: #context manager, close the file automatically

        trkeys = list(trModels.keys())

        # Get number of examples #TODO assert that all groups have the same number of samples
        ntr = len(trModels[trkeys[0]])
        
        # Get shape of examples - assume all examples have the same dimensions
        xshape = list(trModels[trkeys[0]].items())[0][1].shape # NOTE: Groups dont have a shape or type, datasets do. Search shape info in a dataset.  
        
        # Allocate output arrays  
        Ref_models    = np.empty((ntr,xshape[0],xshape[1]),dtype='float32')
        True_models   = np.empty((ntr,xshape[0],xshape[1]),dtype='float32')
        Smooth_models = np.empty((ntr,xshape[0],xshape[1]),dtype='float32')
        Img_models    = np.empty((ntr,xshape[0],xshape[1]),dtype='float32')
        
        for group in trModels.keys(): #TODO this is very ugly, pythonize it! 
            i=0
            if group == 'reflectivity':
                for dset in trModels[group].keys() :
                    Ref_models[i, :, :]  = trModels[group][dset][:] # adding [:] returns a numpy array    
                    i+=1
            elif group == 'velocity':
                for dset in trModels[group].keys() :
                    True_models[i, :, :] = trModels[group][dset][:] # adding [:] returns a numpy array    
                    i+=1
            elif group == 'velocity_mig':
                for dset in trModels[group].keys() :
                    Smooth_models[i, :, :]   = trModels[group][dset][:] # adding [:] returns a numpy array    
                    i+=1
            elif group == 'image':
                for dset in trModels[group].keys() :
                    Img_models[i, :, :]   = trModels[group][dset][:] # adding [:] returns a numpy array    
                    i+=1
    return Ref_models, Img_models, True_models, Smooth_models 
