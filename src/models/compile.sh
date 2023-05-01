rm *.so *.c
mv __init__.py init_tmp.py
python setup.py build_ext --inplace
cp ./build/lib.linux-x86_64-3.9/src/models/CythonModel.cpython-39-x86_64-linux-gnu.so .
cp ./build/lib.linux-x86_64-3.9/src/models/CythonSVD.cpython-39-x86_64-linux-gnu.so .
rm -rf build
mv init_tmp.py __init__.py
