TARGET = dforest/_dtree.cpython-36m-x86_64-linux-gnu.so
OBJS = dforest/_dtree.pyx dforest/dtree-cpp/*.cpp dforest/dtree-cpp/*.h

$(TARGET): $(OBJS)
	python setup.py build_ext -if

clean:
	rm -f $(TARGET)
