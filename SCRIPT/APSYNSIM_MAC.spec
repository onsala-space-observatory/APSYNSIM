# -*- mode: python -*-

# SPEC FILE FOR PYINSTALLER:

# RUN PYINSTALLER WITH THE -w FLAG!!

a = Analysis(['./SCRIPT/APSYNSIM.py'],
             pathex=['./APSINSYM_MAC'],
             hiddenimports=['scipy.special._ufuncs_cxx','mpl_toolkits','mpl_toolkits.mplot3d'],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='APSYNSIM.app',
          debug=False,
          strip=None,
          upx=True,
          console=True , icon='./COMPILE/APSYNSIM_icon_small.ico')

import mpl_toolkits.mplot3d
import os


coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=None,
               upx=True,
               name='APSYNSIM')

