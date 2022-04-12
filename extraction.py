import nScopePy as ns

from ctypes import CDLL, Structure, POINTER, c_int, c_uint, c_float, c_double, c_bool, cast, byref
import os
import time

# Definitions for using lib_gpu via FFI
class overclock_setting(Structure):
  _fields_ = [
    ('editable', c_bool),
    ('currentValue', c_float),
    ('minValue', c_float),
    ('maxValue', c_float),
  ]


class overclock_profile(Structure):
  _fields_ = [
    ('core', overclock_setting),
    ('memory', overclock_setting),
    ('shader', overclock_setting),
    ('overvolt', overclock_setting),
  ]

nvidia = CDLL(os.path.join(os.path.dirname(__file__), "../../lib_gpu/Release/lib_gpu.dll"))
nvidia.init_simple_api.restype = c_bool
nvidia.get_overclock_profile.restype = overclock_profile
nvidia.get_overclock_profile.argtypes = [c_uint]
nvidia.overclock.restype = c_bool
nvidia.overclock.argtypes = [c_uint, c_uint, c_float]

obj = ns.nScopeObj()

# Read value as percent
def read_pct():
  # We use an average of 10 values... just cause
  x = obj.readCh1(10,10)
  avg = float(sum(x))/len(x)
  return round((avg/5)*20)/20

if nvidia.init_simple_api():
  # The highest Core MHz adjustment, we'll sweep from 0 to this value.
  max_core = 150
  # Number of polls, just to ensure we don't run infinitely
  max_num = 25
  i = 0

  while i < max_num:
    pct = read_pct()
    core = max_core * pct
    print 'Percent:', pct*100
    print 'Core:', core, "MHz"
    if not nvidia.overclock(0, 0, core):
      print "Error overclocking"
      break
    p = nvidia.get_overclock_profile(0)
    print "Core actually:", p.core.currentValue, "MHz"
    time.sleep(.5)
    i += 1
else:
  print "NO NVIDIA???"
