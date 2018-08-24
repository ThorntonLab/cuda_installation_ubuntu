Key:

```
pip3 install tensorflow_gpu
```

* https://askubuntu.com/questions/1028830/how-do-i-install-cuda-on-ubuntu-18-04

The relevant instructions from above are:

```
sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt update

sudo ubuntu-drivers autoinstall

sudo reboot

sudo apt install nvidia-cuda-toolkit

sudo apt-get -f intall cuda-9-2
```

The above commands lead to a runtime error:

```
import tensorflow as tf
```

which gives:

```
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.5/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.5/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory


Failed to load the native TensorFlow runtime.
```

Solution:

```
sudo apt-get -f install cuda-9-0
```

That led to missing:

```
ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory
```

Solution: register with NVIDIA and download .deb pacakges for CUDA 9.0.  I got both dev and runtime.


* https://www.tensorflow.org/guide/using_gpu

This link has code for how to test if GPU is working:

```
import tensorflow as tf
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
