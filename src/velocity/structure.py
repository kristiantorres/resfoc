import numpy as np

def vel_structure(n):
  """
  Returns a dictionary of parameters that contain information
  for recreating the structure of the velocity model.
  This dictionary is necessary to retain if multi-parameter
  models need to be created
  """
  outdict = {}
  ### First deposit
  ## Deposit
  thick1min = 50
  thick1max = 100
  thick1 = np.random.rand()*(thick1max - thick1min) + thick1min
  thick1 = int(thick1)
  outdict['thick1'] = thick1

  ## Large folding of the deposit
  amp1min = 900
  amp1max = 4500
  amp1 = np.random.rand()*(amp1max - amp1min) + amp1min
  outdict['amp1'] = amp1
  # Azimuth
  outdict['az1'] = np.random.rand()*360.0
  # Wavelength
  wav1max = 0.7
  wav1min = 0.01
  outdict['wav1'] = np.random.rand()*(wav1max - wav1min) + wav1min

  ## Fault the folded layers. Lots of faulting and large faults
  outdict['nfaults1'] = np.random.choice(range(20))
  begzmax=0.4
  begzmin=0.6
  outdict['begzf1'] = np.random.rand()*(begzmax - begzmin) + begzmin
  # Make the faulting all in the same random azimuth
  outdict['faz1'] = np.random.rand()*360.0
  # Large fault throw
  outdict['theta_shift1'] = 7.0

  ### Second thin set of layers (covers the faults)
  ## Deposit
  thick2min = 50
  thick2max = 100
  outdict['thick2'] = int(np.random.rand()*(thick2max - thick2min) + thick2min)
  #outdict['thick2'] = 450 - thick1

  ## Gentle folding
  # Amplitude
  amp2min = 500
  amp2max = 2000
  outdict['amp2'] = np.random.rand()*(amp2max - amp2min) + amp2min
  # Azimuth
  outdict['az2'] = np.random.rand()*360.0
  # Wavelength
  wav2max = 0.7
  wav2min = 0.07
  outdict['wav2'] = np.random.rand()*(wav2max - wav2min) + wav2min

  ## Faulting
  outdict['nfaults2'] = np.random.choice(range(5))
  begzmax=0.3
  begzmin=0.4
  outdict['begzf2'] = np.random.rand()*(begzmax - begzmin) + begzmin
  # Faulting in same azimuth
  outdict['faz2'] = np.random.rand()*360.0
  # Smaller fault throw
  outdict['theta_shift2'] = 4.0

  ### Third set of layers
  # Deposit
  thick3min = 50
  thick3max = 100
  outdict['thick3'] = int(np.random.rand()*(thick3max - thick3min) + thick3min)

  ## Gentle folding
  # Amplitude
  amp3min = 150
  amp3max = 2000
  outdict['amp3'] = np.random.rand()*(amp3max - amp3min) + amp3min
  # Azimuth
  outdict['az3'] = np.random.rand()*360.0
  # Wavelength
  wav3max = 0.7
  wav3min = 0.07
  outdict['wav3'] = np.random.rand()*(wav3max - wav3min) + wav3min

  ## Faulting
  outdict['nfaults3'] = np.random.choice(range(5))
  begzmax=0.3
  begzmin=0.4
  outdict['begzf3'] = np.random.rand()*(begzmax - begzmin) + begzmin
  # Faulting in same azimuth
  outdict['faz3'] = np.random.rand()*360.0
  # Smaller fault throw
  outdict['theta_shift3'] = 4.0

  ### Fourth set of layers
  ## Deposit
  thick4min = 50
  thick4max = 100
  outdict['thick4'] = int(np.random.rand()*(thick4max - thick4min) + thick4min)

  ## Gentle folding
  # Amplitude
  amp4min = 150
  amp4max = 2000
  outdict['amp4'] = np.random.rand()*(amp4max - amp4min) + amp4min
  # Azimuth
  outdict['az4'] = np.random.rand()*360.0
  # Wavelength
  wav4max = 0.7
  wav4min = 0.07
  outdict['wav4'] = np.random.rand()*(wav4max - wav4min) + wav4min

  ## Faulting
  outdict['nfaults4'] = np.random.choice(range(5))
  begzmax=0.2
  begzmin=0.1
  outdict['begzf4'] = np.random.rand()*(begzmax - begzmin) + begzmin
  # Faulting in same azimuth
  outdict['faz4'] = np.random.rand()*360.0
  # Smaller fault throw
  outdict['theta_shift4'] = 4.0

  ## Thickness of water layer
  outdict['wthick'] = 10

  ## Slicing
  outdict['choice'] = np.random.choice([0,1])
  outdict['idx'] = np.random.choice(range(n))

  return outdict

